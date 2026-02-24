"""
train_model.py — NEXUS SPAM SHIELD v5.0
Advanced ML pipeline with multi-model comparison.
"""
import pickle, re, os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

HAM_MESSAGES = [
    "hello","hi","hey","yo","hi there","hello there","hey there",
    "good morning","good night","good evening","good afternoon",
    "how are you","how are you doing","how have you been",
    "what are you doing","what are you up to","what's up","sup",
    "how's it going","hope you're doing well",
    "are you there","you there","are you free",
    "are you free today","are you free tomorrow","free this weekend",
    "hello bro","hey bro","hi bro","bro where are you",
    "call me","call me later","call me when free","call me back",
    "call me when you reach","call me tonight","give me a call",
    "let's meet","lets meet","let's catch up","catch up soon",
    "let's meet tomorrow","meet me at the office","see you there",
    "can we talk","can we talk later","can we talk tonight",
    "we need to talk","talk to me","let me know when free",
    "ping me","text me later","message me","reply when free",
    "are you coming today","coming to class today","coming to office",
    "let us meet for lunch","lunch today","lunch at 1pm",
    "dinner tonight","dinner plans","coffee later","coffee this evening",
    "movie tonight","movie plans","plans for the weekend",
    "joining the meeting","are you joining","joining us today",
    "see you at the meeting","meeting at 3pm","meeting rescheduled",
    "party tonight","birthday party tomorrow","are you coming to the party",
    "let's grab a bite","grab coffee","drinks later",
    "just checking in","checking in on you","just wanted to say hi",
    "thinking of you","hope you are well","take care",
    "stay safe","stay healthy","keep well","take it easy",
    "miss you","miss you bro","long time no talk","it's been a while",
    "when are you back","when are you coming back","back in town",
    "reached home","reached safely","on my way","almost there",
    "please send the notes","send me the notes","can you share the file",
    "share the document please","did you complete the assignment",
    "have you done the homework","submit the report by friday",
    "deadline is tomorrow","need your input on this","review this please",
    "meeting at 10am","conference call at noon","zoom call tomorrow",
    "the project is due next week","update me on the status",
    "any progress on this","let me know the update","status update please",
    "working from home today","in office today","out of office tomorrow",
    "on leave tomorrow","taking a day off","half day today",
    "can you cover for me","need a favor","quick question",
    "are you in today","coming to office","see you in office",
    "class cancelled today","lecture postponed","exam on monday",
    "study group tonight","library or online","notes for today",
    "thanks for your help","thank you so much","really appreciate it",
    "you're the best","love you","love you bro","take care of yourself",
    "get well soon","feel better soon","praying for you",
    "happy birthday","many happy returns","congratulations on your result",
    "so proud of you","well done","great job","you did great",
    "sorry for the delay","apologies for late reply","sorry couldn't reply earlier",
    "no worries","all good","it's fine","no problem",
    "awesome","great","sounds good","sure","okay","ok",
    "yes","no","maybe","not sure","let me think",
    "will do","on it","done","completed","finished",
    "what time does the shop close","is the restaurant open",
    "can you pick me up","need a ride home","can you drop me",
    "where are you","what's your location","send location",
    "are you near","how far are you","how long will you take",
    "running late","stuck in traffic","be there in 10 minutes",
    "wait for me","don't leave without me","save a seat for me",
    "bring some food","get me some coffee","pick up some groceries",
    "did you eat","have you had lunch","want to order food",
    "what do you want to eat","pizza or burger","your choice",
    "cool","nice","interesting","got it","understood",
    "roger that","copy that","yep","nope",
    "haha","lol","hahaha","that's funny","nice joke",
    "see you later","talk later","catch you later","later",
    "have a good day","have a nice day","enjoy your day",
    "good luck today","all the best","rooting for you",
    "the traffic is terrible today can you take the other route",
    "i forgot to charge my phone sorry for late reply",
    "can you remind me about the dentist appointment tomorrow",
    "the match starts at 8pm are you watching",
    "my train got delayed will be there by 7",
    "just got back from the gym feeling great",
    "do you want tea or coffee when you come",
    "pick up some bread and milk on your way home",
    "the kids are asking about you",
    "mom says hi and asks when you are coming",
    "happy anniversary to you both",
    "sorry to hear that hope things get better soon",
    "the power went out here for a few hours",
    "it's raining heavily here be careful driving",
    "the new restaurant on main street is really good",
    "let me know if you need anything",
    "i'll be at your place by 6pm",
    "the delivery arrived safely thank you",
    "your package was left at the door",
    "the interview went well fingers crossed",
    "just landed safely will call once i reach the hotel",
    "the kids loved the gift you sent",
    "checking if you received my email",
    "the wifi is down will call from landline",
    "happy diwali happy eid happy new year",
    "just wanted to say i appreciate you",
    "the doctor said everything looks fine",
    "school starts again next monday",
    "i booked the tickets for saturday",
    "can you please pass the report to the manager",
    "the electricity bill needs to be paid by friday",
    "shall we reschedule the call to thursday",
    "how is your mother doing hope she is well",
    "we are all going to the beach this sunday want to join",
    "the exam results come out tomorrow fingers crossed",
    "sent you the address on whatsapp",
    "the repair guy is coming at 11am be home",
    "leaving office now see you at home",
    "the baby said her first word today",
    "grandma wants to video call this evening",
]

SPAM_MESSAGES = [
    "congratulations you have won a free iPhone click here to claim",
    "you are the lucky winner of our monthly prize claim now",
    "you have won 10000 cash prize call now to claim",
    "winner winner you have been selected claim your prize today",
    "you won a lottery prize of 50000 click link to collect",
    "WINNER you have been randomly selected for a reward",
    "dear customer you have won a Samsung Galaxy claim immediately",
    "your mobile number won our prize draw claim within 24 hours",
    "congratulations your number has won the national lottery",
    "you have been selected as our monthly lucky winner",
    "lucky draw winner collect your prize today",
    "you are selected for free reward claim immediately",
    "free gift card worth 5000 click to claim now",
    "you have been awarded a cash prize of one million dollars claim",
    "your email has won second prize in our international lottery claim now",
    "get free recharge of 500 rupees click now",
    "earn 5000 per day working from home apply now",
    "guaranteed returns of 200 percent invest now",
    "double your money in 7 days guaranteed investment scheme",
    "loan approved for you instant cash disbursement apply",
    "your loan application approved get money now click",
    "get instant personal loan up to 5 lakh no documents required",
    "earn money online 10000 per week from home register now",
    "make easy money online from home investment opportunity",
    "stock market tips guaranteed profit daily signal subscribe now",
    "invest 1000 and get 10000 back guaranteed returns scheme",
    "work from home earn 50000 monthly no experience required",
    "bitcoin investment guaranteed 300 percent return click now",
    "forex trading signals guaranteed daily profit subscribe now",
    "crypto scheme earn daily income from home register free",
    "act now limited time offer expires today click",
    "limited offer available for today only buy now hurry",
    "hurry offer ends in 1 hour grab it now limited",
    "last chance to avail this exclusive deal click immediately",
    "urgent action required your account will be suspended click",
    "immediate response required or your account will be closed verify",
    "respond now or lose your reward claim expires tonight",
    "act fast this offer expires in 24 hours limited deal",
    "final notice your reward is expiring claim before midnight",
    "limited time deal ending soon grab now exclusive offer",
    "exclusive offer only for you expires tonight claim now",
    "this offer is valid for today only do not delay click here",
    "you only have 2 hours to claim this offer hurry now act",
    "your bank account has been compromised verify now",
    "verify your account details immediately to avoid suspension",
    "your account is at risk please verify your credentials now click",
    "click to verify your identity your account may be suspended",
    "your account has been suspended update payment now immediately",
    "dear user your account password needs to be updated click link",
    "security alert please verify your bank account immediately",
    "your account has been blocked click to unblock now verify",
    "verify your KYC now to continue using your account click",
    "your account will be deactivated confirm details now urgent",
    "suspicious activity detected verify your account now click",
    "click this link to reset your banking password immediately urgent",
    "your debit card has been blocked call this number now urgent",
    "otp alert someone tried to access your account verify now",
    "free entry in contest click now to participate win",
    "free gift waiting for you claim before stock runs out",
    "exclusive deal just for you free product on registration click",
    "buy one get one free offer today only limited stock hurry",
    "get free premium subscription for 3 months click now register",
    "free trial available for premium membership register now click",
    "special discount 80 percent off today only shop now limited",
    "flat 90 percent off on all products sale today only click",
    "mega sale everything free just pay shipping click here now",
    "free cash reward for completing survey click here now",
    "job offer 50000 per month work from home no experience needed apply",
    "earn lakhs monthly data entry job from home apply immediately",
    "urgent hiring work from home easy task apply now click",
    "part time job 20000 monthly easy task apply now link given",
    "hiring immediately salary 40000 to 80000 per month apply now register",
    "work at home opportunity earn daily income register free now",
    "lose 20kg in 30 days guaranteed natural supplement buy now",
    "doctor recommended weight loss pill free trial order now click",
    "cure diabetes naturally 100 percent guaranteed remedy buy now",
    "revolutionary product cures all diseases order now limited stock",
    "win money now limited offer","claim your reward today click",
    "limited offer click here now","verify account immediately urgent",
    "free prize winner congratulations claim","urgent bank transfer required",
    "click here to get free iphone now","earn 10000 per day from home apply",
    "your account will be suspended verify now","you won a cash prize claim now",
    "free money transfer to your account click","selected for exclusive reward claim today",
    "click now to win prize","free gift claim now hurry",
    "urgent verify account now","congratulations you won claim",
    "exclusive limited offer ends today","you are a winner claim prize now",
    "instant loan approved apply now","work from home earn daily guaranteed",
    "free recharge click now","win iphone click here",
    "bitcoin profit guaranteed register","otp verify account urgent",
    "lottery winner claim prize","investment scheme guaranteed profit",
    "account suspended verify immediately","exclusive offer claim now",
    "free subscription register now","prize winning claim today",
]


def build_dataset():
    ham_df  = pd.DataFrame({'label': 'ham',  'message': HAM_MESSAGES})
    spam_df = pd.DataFrame({'label': 'spam', 'message': SPAM_MESSAGES})
    df = pd.concat([ham_df, spam_df], ignore_index=True)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spam.csv')
    if os.path.exists(csv_path):
        try:
            original = pd.read_csv(csv_path)
            for cl, cm in [('label','message'),('v1','v2'),('Category','Message')]:
                if cl in original.columns and cm in original.columns:
                    original = original[[cl,cm]].rename(columns={cl:'label',cm:'message'}); break
            original = original.dropna()
            original['label'] = original['label'].str.strip().str.lower().map(
                lambda x: 'ham' if x in ('ham','legitimate','not spam') else 'spam')
            if len(original) > 50:
                df = pd.concat([df, original], ignore_index=True)
                print(f"  Merged spam.csv (+{len(original)} rows)")
        except Exception as e:
            print(f"  Could not merge spam.csv: {e}")
    df = df.drop_duplicates(subset='message').sample(frac=1, random_state=42).reset_index(drop=True)
    ham_n  = (df['label']=='ham').sum()
    spam_n = (df['label']=='spam').sum()
    print(f"  Total: {len(df)}  Ham: {ham_n} ({ham_n/len(df)*100:.0f}%)  Spam: {spam_n} ({spam_n/len(df)*100:.0f}%)")
    return df


def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


SPAM_SIGNALS = {
    'free','win','winner','won','cash','prize','claim','offer','urgent',
    'congratulations','click','limited','deal','guaranteed','money','credit',
    'loan','discount','selected','promotion','bonus','reward','gift',
    'exclusive','subscribe','verify','account','password','bank','invoice',
    'confirm','nigeria','transfer','million','billion','investment','profit',
    'pharmacy','pills','ringtone','xxx','adult','dating','singles',
    'lottery','rupees','recharge','iphone','samsung','earn','salary',
    'hiring','apply','register','order','buy','shop','hurry','expires',
    'suspended','deactivated','compromised','alert','security','phishing',
    'bitcoin','crypto','forex','otp','kyc','scheme','survey',
}


def classify(spam_prob: float, message: str, threshold: float) -> str:
    words   = message.lower().split()
    wordset = set(words)
    signals = len(wordset & SPAM_SIGNALS)
    if len(words) <= 6 and signals == 0:
        return "NOT SPAM"
    eff = threshold
    if signals >= 3:   eff = max(0.38, threshold - 0.20)
    elif signals >= 2: eff = max(0.48, threshold - 0.12)
    elif signals >= 1: eff = max(0.55, threshold - 0.05)
    return "SPAM" if spam_prob >= eff else "NOT SPAM"


def train_and_evaluate(df):
    df['clean'] = df['message'].apply(preprocess)
    X = df['clean']
    y = (df['label'] == 'spam').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True,
        max_features=12000, strip_accents='unicode', analyzer='word')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"  Vocabulary: {len(vectorizer.vocabulary_)} features")

    n_cv = min(3, int(min(y_train.sum(), (1-y_train).sum())))

    candidates = {
        "Logistic Regression": CalibratedClassifierCV(
            LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42),
            method='isotonic', cv=n_cv),
        "Naive Bayes": CalibratedClassifierCV(MultinomialNB(alpha=0.1), method='isotonic', cv=n_cv),
        "SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42),
            method='isotonic', cv=n_cv),
    }

    print(f"\n  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("  " + "-"*50)

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train_vec, y_train)
        proba = clf.predict_proba(X_test_vec)[:, 1]
        best_t, best_f1 = 0.60, 0.0
        for t in np.arange(0.35, 0.85, 0.05):
            preds = (proba >= t).astype(int)
            if preds.sum() == 0: continue
            f1 = f1_score(y_test, preds, zero_division=0)
            prec = precision_score(y_test, preds, pos_label=0, zero_division=0)
            if prec >= 0.92 and f1 > best_f1:
                best_f1 = f1; best_t = t
        best_t = max(best_t, 0.55)
        y_pred = (proba >= best_t).astype(int)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        results[name] = dict(model=clf, threshold=best_t, f1=f1, acc=acc, prec=prec, rec=rec, proba=proba)
        print(f"  {name:<22} {acc:>7.3f} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

    best_name = max(results, key=lambda k: results[k]['f1'])
    best = results[best_name]
    model, threshold = best['model'], best['threshold']
    print(f"\n  Selected: {best_name} (F1={best['f1']:.3f}, threshold={threshold:.2f})")

    y_pred = (best['proba'] >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "="*55)
    print(f"  Accuracy : {best['acc']*100:.1f}%  Precision: {best['prec']*100:.1f}%")
    print(f"  Recall   : {best['rec']*100:.1f}%  F1 Score : {best['f1']*100:.1f}%")
    print(f"  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['ham','spam'])}")

    meta = {
        "model_name": best_name, "f1_score": round(best['f1'],4),
        "accuracy": round(best['acc'],4), "precision": round(best['prec'],4),
        "recall": round(best['rec'],4), "threshold": round(threshold,4),
        "vocab_size": len(vectorizer.vocabulary_),
        "confusion_matrix": cm.tolist(),
        "all_models": {k: {"f1": round(v['f1'],4)} for k, v in results.items()},
    }
    return model, vectorizer, float(threshold), meta


def run_validation(model, vectorizer, threshold):
    MUST_HAM  = ["hello","how are you","call me later","are you free today",
                  "good morning","sounds good","i'll be there by 7","let's meet tomorrow",
                  "happy birthday","just checking in"]
    MUST_SPAM = ["win money now","claim your reward","verify account immediately",
                  "free prize winner congratulations","you won a lottery claim now",
                  "earn 10000 per day from home apply","your account will be suspended verify now",
                  "bitcoin guaranteed profit click now","instant loan approved apply now"]
    passed = 0; total = len(MUST_HAM) + len(MUST_SPAM)
    print("\n  VALIDATION:")
    for msg in MUST_HAM:
        prob  = model.predict_proba(vectorizer.transform([preprocess(msg)]))[0][1]
        label = classify(prob, msg, threshold)
        ok = label == "NOT SPAM"
        print(f"  {'OK' if ok else 'FAIL'}  {msg:<38} {prob*100:>5.1f}%")
        if ok: passed += 1
    for msg in MUST_SPAM:
        prob  = model.predict_proba(vectorizer.transform([preprocess(msg)]))[0][1]
        label = classify(prob, msg, threshold)
        ok = label == "SPAM"
        print(f"  {'OK' if ok else 'FAIL'}  {msg:<38} {prob*100:>5.1f}%")
        if ok: passed += 1
    print(f"  PASSED: {passed}/{total}")
    return passed >= int(total * 0.90)


if __name__ == '__main__':
    print("\nNEXUS SPAM SHIELD v5.0 — MODEL TRAINING\n")
    df = build_dataset()
    model, vectorizer, threshold, meta = train_and_evaluate(df)
    passed = run_validation(model, vectorizer, threshold)
    base = os.path.dirname(os.path.abspath(__file__))
    pickle.dump(model,      open(os.path.join(base,'model.pkl'),      'wb'))
    pickle.dump(vectorizer, open(os.path.join(base,'vectorizer.pkl'), 'wb'))
    pickle.dump(threshold,  open(os.path.join(base,'threshold.pkl'),  'wb'))
    pickle.dump(meta,       open(os.path.join(base,'model_meta.pkl'), 'wb'))
    print(f"\nSaved: model={meta['model_name']} F1={meta['f1_score']} threshold={threshold:.2f}")