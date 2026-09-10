"""Template-based multilingual synthetic scam data + hard negatives.

Zero-data categories (otp_request, payment_pressure, investment_crypto, ...)
get seeded coverage across Tier 1 + Tier 2 languages WITHOUT an LLM API:
scam SMS are formulaic, so per-language template banks with slot filling
(otp codes, amounts, brands, links) produce realistic training signal.
Hard negatives (legitimate bank OTPs, parcel updates, receipts) attack the
false-positive problem in the same languages.

Honest scope: this is the seed layer - formulaic but real-pattern signal.
Phase 3 upgrades naturalness with LLM-generated data; the pipeline wiring
never changes. Every row is tagged source="synthetic_ml_{lang}" so reports
show exactly how much synthetic signal is in the mix.
"""

import random

from scamless.data.schemas import make_message

SOURCE_PREFIX = "synthetic_ml"

LANGS = {
    "en": {"cur": "$"},
    "hinglish": {"cur": "₹"},
    "hi": {"cur": "₹"},
    "bn": {"cur": "₹"},
    "mr": {"cur": "₹"},
    "pa": {"cur": "₹"},
    "gu": {"cur": "₹"},
    "ta": {"cur": "₹"},
    "te": {"cur": "₹"},
    "es": {"cur": "$"},
    "pt": {"cur": "R$"},
    "fr": {"cur": "€"},
    "ar": {"cur": "$"},
    "id": {"cur": "Rp"},
}

CATEGORIES = [
    "otp_request",
    "payment_pressure",
    "investment_crypto",
    "lottery_prize",
    "job_task",
    "gov_bank_impersonation",
    "delivery_scam",
    "advance_fee",
    "account_suspension",
]

BANKS = ["SBI", "HDFC", "ICICI", "Axis Bank", "PayPal", "Chase", "Wells Fargo", "Barclays", "Axis", "Paytm"]
APPS = ["WhatsApp", "Telegram", "Instagram", "Facebook", "Google Pay", "PhonePe"]
LINKS = [
    "secure-verify-login.com",
    "account-alerts.info",
    "pay-verify.xyz",
    "kyc-update-online.com",
    "track-parcel-status.info",
    "wallet-refund-desk.com",
]

# ---------------------------------------------------------------------------
# Hard negatives: legitimate messages that LOOK scammy (the FP killer)
# ---------------------------------------------------------------------------
HARD_NEGATIVES = {
    "en": [
        "Your {bank} OTP for login is 4482. Valid 10 min. Do not share with anyone.",
        "Dispatch alert: parcel DL-{code} is out for delivery, arriving today 2-6 PM. Track: courier.example/track",
        "Payment of {amount} to GreenMart Supermarket was successful. Balance available on request. No action needed.",
        "{bank}: Your salary of {amount} has been credited to account XX{code}. This is an automated alert.",
        "Reminder: appointment with Dr. Sharma tomorrow at 11:30 AM at City Clinic. Reply 1 to confirm.",
        "Your electricity bill of {amount} was auto-debited successfully. Receipt: bill.example/{code}",
    ],
    "hinglish": [
        "Aapke {bank} account me {amount} credit hue hain. Yeh ek automatic alert hai, is par koi action nahi chahiye.",
        "Aapka delivery parcel DL-{code} aaj shaam tak pahunch jayega. Thank you for shopping with us.",
        "{bank} OTP 9921 aapke login ke liye hai. 10 minute valid. Kisi se share na karein.",
        "Bill payment {amount} successful. Receipt aapke email par bhej di gayi hai.",
        "Reminder: kal subah 11 baje Dr. Mehta ka appointment hai, City Hospital. Confirm karne ke liye 1 reply karein.",
    ],
    "hi": [
        "आपके {bank} खाते में {amount} जमा हुए हैं। यह स्वचालित सूचना है, कोई कार्रवाई आवश्यक नहीं।",
        "आपका पार्सल DL-{code} आज शाम तक पहुंच जाएगा। ट्रैकिंग के लिए कोई भुगतान आवश्यक नहीं।",
        "{bank} लॉगिन OTP 5531 है। 10 मिनट के लिए वैध। किसी के साथ साझा न करें।",
        "बिजली बिल {amount} का भुगतान सफल रहा। रसीद ईमेल पर भेज दी गई है।",
        "अपॉइंटमेंट रिमाइंडर: कल सुबह 11:30 बजे, सिटी क्लिनिक। पुष्टि करने के लिए 1 भेजें।",
    ],
    "bn": [
        "আপনার {bank} অ্যাকাউন্টে {amount} জমা হয়েছে। এটি একটি স্বয়ংক্রিয় সতর্কতা।",
        "আপনার পার্সেল DL-{code} আজ সন্ধ্যার মধ্যে পৌঁছাবে। কোনো পেমেন্টের প্রয়োজন নেই।",
        "{bank} লগইন OTP 7402। ১০ মিনিটের জন্য বৈধ। কারও সাথে শেয়ার করবেন না।",
        "বিল {amount} পরিশোধ সফল হয়েছে। রিসিট ইমেইলে পাঠানো হয়েছে।",
    ],
    "mr": [
        "तुमच्या {bank} खात्यात {amount} जमा झाले आहेत. ही स्वयंचलित सूचना आहे.",
        "तुमचा पार्सल DL-{code} आज संध्याकाळी पोहोचेल. कोणतीही पेमेंट आवश्यक नाही.",
        "{bank} लॉगिनसाठी OTP 6620 आहे. 10 मिनिटे वैध. कोणासह शेअर करू नका.",
    ],
    "pa": [
        "ਤੁਹਾਡੇ {bank} ਖਾਤੇ ਵਿੱਚ {amount} ਜਮ੍ਹਾਂ ਹੋਏ ਹਨ। ਇਹ ਸਵੈਚਾਲਿਤ ਸੂਚਨਾ ਹੈ।",
        "ਤੁਹਾਡਾ ਪਾਰਸਲ DL-{code} ਅੱਜ ਸ਼ਾਮ ਤੱਕ ਪਹੁੰਚ ਜਾਵੇਗਾ। ਕੋਈ ਭੁਗਤਾਨ ਨਹੀਂ ਚਾਹੀਦਾ।",
        "{bank} ਲੌਗਿਨ ਲਈ OTP 8134 ਹੈ। 10 ਮਿੰਟ ਵੈਧ। ਕਿਸੇ ਨਾਲ ਸਾਂਝਾ ਨਾ ਕਰੋ।",
    ],
    "gu": [
        "તમારા {bank} ખાતામાં {amount} જમા થયા છે. આ સ્વચાલિત સૂચના છે.",
        "તમારું પાર્સલ DL-{code} આજે સાંજ સુધીમાં પહોંચશે. કોઈ ચૂકવણી જરૂરી નથી.",
        "{bank} લોગિન માટે OTP 5971 છે. 10 મિનિટ માન્ય. કોઈ સાથે શેર ન કરો.",
    ],
    "ta": [
        "உங்கள் {bank} கணக்கில் {amount} செலுத்தப்பட்டது. இது தானியங்கி அறிவிப்பு.",
        "உங்கள் பொதி DL-{code} இன்று மாலை வந்து சேரும். பணம் செலுத்த வேண்டியதில்லை.",
        "{bank} உள்நுழைவு OTP 3389. 10 நிமிடங்கள் செல்லுபடி. யாருடனும் பகிர வேண்டாம்.",
    ],
    "te": [
        "మీ {bank} ఖాతాలో {amount} జమ అయ్యాయి. ఇది ఆటోమేటిక్ నోటిఫికేషన్.",
        "మీ పార్సెల్ DL-{code} ఈ రోజు సాయంత్రం వచ్చేస్తుంది. చెల్లింపు అవసరం లేదు.",
        "{bank} లాగిన్ OTP 7745. 10 నిమిషాలు చెల్లుబాటు. ఎవరితోనూ పంచకండి.",
    ],
    "es": [
        "Su banco {bank} le informa: pago de {amount} realizado con éxito. No necesita hacer nada.",
        "Su paquete DL-{code} llega hoy entre las 14:00 y las 18:00. No se requiere pago.",
        "OTP 6153 para iniciar sesion en {bank}. Valido 10 minutos. No lo comparta.",
        "Recordatorio: cita medica manana a las 11:30 en Clinica Central. Responda 1 para confirmar.",
    ],
    "pt": [
        "O banco {bank} informa: pagamento de {amount} realizado com sucesso.",
        "Sua encomenda DL-{code} chega hoje entre 14h e 18h. Nao e necessario pagar.",
        "OTP 9271 para acessar o {bank}. Valido por 10 minutos. Nao compartilhe.",
    ],
    "fr": [
        "Votre banque {bank} vous informe : paiement de {amount} effectue avec succes.",
        "Votre colis DL-{code} sera livre aujourd'hui entre 14h et 18h. Aucun paiement requis.",
        "OTP 4817 pour vous connecter a {bank}. Valable 10 minutes. Ne le partagez pas.",
    ],
    "ar": [
        "يرجى العلم من {bank} أن الدفع بقيمة {amount} تم بنجاح. لا حاجة لأي إجراء.",
        "طرودكم DL-{code} سيصل اليوم بين 2 و 6 مساءً. لا يلزم أي دفع.",
        "رمز التحقق 5294 لتسجيل الدخول إلى {bank}. صالح لمدة 10 دقائق. لا تشاركه مع أحد.",
    ],
    "id": [
        "Bank {bank} menginformasikan: pembayaran {amount} berhasil. Tidak perlu tindakan apa pun.",
        "Paket Anda DL-{code} tiba hari ini antara pukul 14.00-18.00. Tidak perlu bayar.",
        "OTP 8036 untuk masuk ke {bank}. Berlaku 10 menit. Jangan bagikan ke siapa pun.",
    ],
}

# ---------------------------------------------------------------------------
# Scam templates per language per category. {slots} get filled with
# numbers/brands/links so each template yields dozens of variants.
# ---------------------------------------------------------------------------
TEMPLATES = {
    "en": {
        "otp_request": [
            "{bank}: OTP {otp} requested for payment of {amount}. If this was not you, call immediately.",
            "Share OTP {otp} to complete your {app} verification. Do not share this code with anyone from support.",
            "Your verification code is {otp}. Enter it within 5 minutes to avoid account closure.",
        ],
        "payment_pressure": [
            "FINAL WARNING: {amount} pending on your card. Pay now via {link} or legal action starts within {hours} hours.",
            "Your outstanding loan of {amount} is overdue. Clear it today or agents will visit your address.",
            "Immediate payment of {amount} required to avoid court case. Pay via UPI to the number in this message.",
        ],
        "investment_crypto": [
            "Guaranteed returns: turn {amount} into 10x in 30 days with our AI trading platform. Join {link}",
            "Crypto insider signal - last chance. Investors doubled money last week. Deposit {amount} now.",
            "Limited slots: government-approved doubling scheme. {amount} becomes {amount}x2 in {hours} days. Contact agent.",
        ],
        "lottery_prize": [
            "CONGRATULATIONS! You won {amount} in the international lottery. Claim within {hours} hours at {link}",
            "Your mobile number was selected for a {amount} prize. Send your details to claim agent now.",
            "Winner alert: claim your {amount} prize before it goes to the next candidate. Processing fee applies.",
        ],
        "job_task": [
            "Earn {amount} daily doing simple tasks from home. Register at {link} - registration fee {amount} only.",
            "Part-time job: {amount} per day, no experience. Pay small deposit to start today. Limited positions.",
            "Congratulations, you are selected for a work-from-home job. Pay {amount} security fee to receive tasks.",
        ],
        "gov_bank_impersonation": [
            "{bank} ALERT: Your KYC is expired. Update immediately at {link} or your account will be frozen.",
            "RBI notice: verify your bank account details today at {link} to keep your funds active.",
            "This is your bank manager. Confirm your card details on {link} to unblock your {amount} transaction.",
        ],
        "delivery_scam": [
            "Your parcel is held at customs. Pay customs fee of {amount} at {link} to release delivery.",
            "Delivery failed - address incomplete. Reschedule and pay {amount} redirection fee at {link}",
            "Parcel DL-{code} requires a {amount} clearance charge before delivery. Pay today or it returns to sender.",
        ],
        "advance_fee": [
            "Dear friend, I am a diplomat with a consignment box of {amount}. Pay transfer fee {amount} to receive it.",
            "You have inherited {amount} from a late foreign client. Pay legal processing fee to release the funds.",
            "Charity donation of {amount} approved for you. Send bank details plus {amount} transfer charges.",
        ],
        "account_suspension": [
            "Your account will be suspended in {hours} hours due to suspicious activity. Verify at {link} now.",
            "URGENT: unusual login detected. Confirm your password within {hours} hours or lose access permanently.",
            "Final notice: account closure process started. Stop it by updating billing at {link}",
        ],
    },
    "hinglish": {
        "otp_request": [
            "{bank} se OTP {otp} {amount} ke payment ke liye aaya hai. Agar aapne nahi kiya to turant call karein.",
            "OTP {otp} share karein aur {app} verification complete karein. Support wale kabhi OTP nahi maangte.",
            "Aapka verification code {otp} hai. 5 minute me daalein warna account band ho jayega.",
        ],
        "payment_pressure": [
            "LAST WARNING: {amount} pending hai aapke card par. Aaj {link} par pay karein warna case ho jayega.",
            "Loan EMI {amount} overdue hai. Aaj clear karein ya agent ghar aayega.",
            "{amount} turant chukwana zaroori hai warna court case shuru. UPI par pay karein.",
        ],
        "investment_crypto": [
            "Guaranteed returns: {amount} ko 30 din me 10x banayein. Aaj {link} par join karein.",
            "Crypto insider signal - last chance. Pichhle hafte investors ka paisa double hua. {amount} deposit karein.",
            "Limited slots: paisa double scheme, sarkar approved. {amount} ban jayega double {hours} din me.",
        ],
        "lottery_prize": [
            "BADHAI HO! Aapne {amount} jeeta hai international lottery me. {hours} ghante me claim karein {link} par.",
            "Aapka number {amount} ke prize ke liye select hua hai. Details agent ko bhejein aur claim karein.",
            "Winner alert: {amount} ka prize aaj hi claim karein, processing fee lagegi.",
        ],
        "job_task": [
            "Ghar baithe {amount} rozana kamayein, simple tasks. {link} par register karein, sirf {amount} fee.",
            "Part-time job: roz {amount}, experience nahi chahiye. Chhota deposit dekar aaj shuru karein.",
            "Congratulations, aap work-from-home job ke liye select hue hain. {amount} security fee bhejein.",
        ],
        "gov_bank_impersonation": [
            "{bank} ALERT: KYC expire ho gayi hai. Turant {link} par update karein warna account freeze.",
            "RBI notice: bank details aaj {link} par verify karein ya funds block ho jayenge.",
            "Main bank manager bol raha hoon. Card details {link} par confirm karein, {amount} ka transaction unblock hoga.",
        ],
        "delivery_scam": [
            "Parcel customs me ruka hai. {amount} ka fee {link} par pay karein, delivery hogi.",
            "Delivery fail - address adhura hai. {amount} redirection fee dekar {link} par reschedule karein.",
            "Parcel DL-{code} ko {amount} clearance charge chahiye. Aaj pay karein warna wapas jayega.",
        ],
        "advance_fee": [
            "Dost, mere paas {amount} ka consignment box hai. Transfer fee {amount} bhejein, box aapka.",
            "Aapko {amount} ka inheritance mila hai. Legal fee bhejein, fund release hoga.",
            "Charity donation {amount} approved. Bank details aur {amount} charges bhejein.",
        ],
        "account_suspension": [
            "Account {hours} ghante me suspend hoga, suspicious activity ki wajah se. Abhi {link} par verify karein.",
            "URGENT: anjaan login detect hua. {hours} ghante me password confirm karein ya access chalega.",
            "Final notice: account band hone wala hai. {link} par billing update karke rok lein.",
        ],
    },
    "hi": {
        "otp_request": [
            "{bank}: {amount} के भुगतान के लिए OTP {otp} का अनुरोध। यदि आपने नहीं किया तो तुरंत कॉल करें।",
            "OTP {otp} साझा करें और {app} सत्यापन पूरा करें। सपोर्ट कभी OTP नहीं मांगता।",
            "आपका सत्यापन कोड {otp} है। खाता बंद होने से बचने के लिए 5 मिनट में दर्ज करें।",
        ],
        "payment_pressure": [
            "अंतिम चेतावनी: कार्ड पर {amount} बकाया है। आज {link} पर भुगतान करें या {hours} घंटे में कानूनी कार्रवाई।",
            "लोन की किश्त {amount} बकाया है। आज चुकाएं या एजेंट घर आएंगे।",
            "कोर्ट केस से बचने के लिए {amount} तुरंत चुकाएं। इस नंबर पर UPI करें।",
        ],
        "investment_crypto": [
            "गारंटीड रिटर्न: {amount} को 30 दिन में 10x बनाएं। आज {link} पर जुड़ें।",
            "क्रिप्टो इनसाइडर सिग्नल - आखिरी मौका। पिछले हफ्ते पैसा दोगुना हुआ। {amount} जमा करें।",
            "सीमित सीटें: पैसा डबल योजना। {amount} {hours} दिन में दोगुना। एजेंट से संपर्क करें।",
        ],
        "lottery_prize": [
            "बधाई हो! आपने {amount} जीता है। {hours} घंटे में {link} पर क्लेम करें।",
            "आपका नंबर {amount} के पुरस्कार के लिए चुना गया है। एजेंट को विवरण भेजें।",
            "विजेता अलर्ट: {amount} का पुरस्कार आज क्लेम करें, प्रोसेसिंग शुल्क लगेगा।",
        ],
        "job_task": [
            "घर बैठे रोज {amount} कमाएं, आसान काम। {link} पर रजिस्टर करें, केवल {amount} शुल्क।",
            "पार्ट-टाइम नौकरी: रोज {amount}, अनुभव नहीं चाहिए। छोटी जमा राशि देकर आज शुरू करें।",
            "बधाई, आप वर्क-फ्रॉम-होम नौकरी के लिए चुने गए हैं। {amount} सुरक्षा शुल्क भेजें।",
        ],
        "gov_bank_impersonation": [
            "{bank} अलर्ट: KYC समाप्त हो गई है। तुरंत {link} पर अपडेट करें या खाता फ्रीज़ होगा।",
            "RBI सूचना: बैंक विवरण आज {link} पर सत्यापित करें या फंड ब्लॉक होंगे।",
            "मैं बैंक मैनेजर बोल रहा हूं। कार्ड विवरण {link} पर कन्फर्म करें, {amount} का लेनदेन अनब्लॉक होगा।",
        ],
        "delivery_scam": [
            "पार्सल कस्टम्स में रुका है। {amount} शुल्क {link} पर भेजें, डिलीवरी होगी।",
            "डिलीवरी विफल - पता अधूरा है। {amount} शुल्क देकर {link} पर पुनर्निर्धारित करें।",
            "पार्सल DL-{code} को {amount} क्लीयरेंस चार्ज चाहिए। आज भेजें या वापस जाएगा।",
        ],
        "advance_fee": [
            "मित्र, मेरे पास {amount} का कंसाइनमेंट बॉक्स है। ट्रांसफर शुल्क {amount} भेजें।",
            "आपको {amount} की विरासत मिली है। कानूनी शुल्क भेजें, राशि जारी होगी।",
            "दान राशि {amount} स्वीकृत। बैंक विवरण और {amount} शुल्क भेजें।",
        ],
        "account_suspension": [
            "खाता {hours} घंटे में निलंबित होगा। अभी {link} पर सत्यापित करें।",
            "तत्काल: अज्ञात लॉगिन मिला। {hours} घंटे में पासवर्ड कन्फर्म करें या पहुंच बंद।",
            "अंतिम नोटिस: खाता बंदी प्रक्रिया शुरू। {link} पर बिलिंग अपडेट करें।",
        ],
    },
    "bn": {
        "otp_request": [
            "{bank}: {amount} পেমেন্টের জন্য OTP {otp} অনুরোধ করা হয়েছে। আপনি না করে থাকলে এখনই কল করুন।",
            "OTP {otp} শেয়ার করে {app} যাচাই সম্পূর্ণ করুন। সাপোর্ট কখনও OTP চায় না।",
        ],
        "payment_pressure": [
            "চূড়ান্ত সতর্কতা: কার্ডে {amount} বকেয়া। আজ {link} এ পরিশোধ করুন নয়তো আদালতের মামলা।",
            "ঋণের কিস্তি {amount} বকেয়া। আজ পরিশোধ করুন নয়তো এজেন্ট বাড়ি আসবে।",
        ],
        "investment_crypto": [
            "নিশ্চিত রিটার্ন: {amount} কে 30 দিনে 10x করুন। আজ {link} এ যোগ দিন।",
            "ক্রিপ্টো সিগন্যাল - শেষ সুযোগ। {amount} জমা দিন, টাকা দ্বিগুণ হবে।",
        ],
        "lottery_prize": [
            "অভিনন্দন! আপনি {amount} জিতেছেন। {hours} ঘন্টায় {link} এ ক্লেইম করুন।",
            "আপনার নম্বর {amount} পুরস্কারের জন্য নির্বাচিত। এখনই এজেন্টের কাছে দাবি করুন।",
        ],
        "job_task": [
            "বাড়িতে বসে প্রতিদিন {amount} আয় করুন। {link} এ রেজিস্টার করুন, মাত্র {amount} ফি।",
            "অভিনন্দন, আপনি ওয়ার্ক-ফ্রম-হোম চাকরির জন্য নির্বাচিত। {amount} সিকিউরিটি ফি পাঠান।",
        ],
        "gov_bank_impersonation": [
            "{bank} সতর্কতা: KYC শেষ হয়ে গেছে। এখনই {link} এ আপডেট করুন নয়তো অ্যাকাউন্ট ফ্রিজ।",
            "RBI নোটিশ: আজ {link} এ ব্যাংক তথ্য যাচাই করুন নয়তো টাকা ব্লক হবে।",
        ],
        "delivery_scam": [
            "পার্সেল কাস্টমসে আটকে আছে। {amount} ফি {link} এ দিন, ডেলিভারি হবে।",
            "ডেলিভারি ব্যর্থ - ঠিকানা অসম্পূর্ণ। {amount} ফি দিয়ে {link} এ পুনর্নির্ধারণ করুন।",
        ],
        "advance_fee": [
            "বন্ধু, আমার কাছে {amount} এর বাক্স আছে। ট্রান্সফার ফি {amount} পাঠান।",
            "আপনি {amount} উত্তরাধিকার পেয়েছেন। আইনি ফি পাঠান, টাকা ছাড়া হবে।",
        ],
        "account_suspension": [
            "সন্দেহজনক কার্যকলাপের কারণে {hours} ঘন্টায় অ্যাকাউন্ট স্থগিত হবে। এখন {link} এ যাচাই করুন।",
            "জরুরি: অজানা লগইন শনাক্ত। {hours} ঘন্টায় পাসওয়ার্ড নিশ্চিত করুন।",
        ],
    },
    "mr": {
        "otp_request": [
            "{bank}: {amount} साठी OTP {otp} विनंती. तुम्हीं नाही केले असेल तर लगेच कॉल करा.",
            "OTP {otp} शेअर करा आणि {app} पडताळणी पूर्ण करा.",
        ],
        "payment_pressure": [
            "शेवटची सूचना: कार्डवर {amount} बाकी. आज {link} वर भरा नाहीतर खटला होईल.",
            "कर्जाची किस्त {amount} थकबाकी आहे. आज भरा नाहीतर एजंट घरी येईल.",
        ],
        "investment_crypto": [
            "खात्रीशीर परतावा: {amount} ला 30 दिवसांत 10x करा. आज {link} वर सामील व्हा.",
            "क्रिप्टो सिग्नल - शेवटची संधी. {amount} भरा, पैसा दुप्पट होईल.",
        ],
        "lottery_prize": [
            "अभिनंदन! तुम्ही {amount} जिंकला आहात. {hours} तासांत {link} वर क्लेम करा.",
            "तुमचा नंबर {amount} बक्षिसासाठी निवडला आहे. आत्ताच एजंटकडे क्लेम करा.",
        ],
        "job_task": [
            "घरी बसून रोज {amount} कमवा. {link} वर नोंदणी करा, फक्त {amount} फी.",
            "अभिनंदन, तुम्ही वर्क-फ्रॉम-होम नोकरीसाठी निवडला आहात. {amount} सुरक्षा फी पाठवा.",
        ],
        "gov_bank_impersonation": [
            "{bank} सूचना: KYC संपली आहे. आत्ता {link} वर अपडेट करा नाहीतर खाते फ्रीझ.",
            "RBI नोटीस: आज {link} वर बँक तपशील पडताळा नाहीतर रक्कम ब्लॉक.",
        ],
        "delivery_scam": [
            "पार्सल सेवाशुल्कात अडकले आहे. {amount} फी {link} वर भरा.",
            "डिलिव्हरी अयशस्वी - पत्ता अपूर्ण. {amount} फी द्या आणि {link} वर पुन्हा वेळ ठरवा.",
        ],
        "advance_fee": [
            "मित्रा, माझ्याकडे {amount} चा डबा आहे. ट्रान्सफर फी {amount} पाठवा.",
            "तुम्हाला {amount} वारसा मिळाला आहे. कायदेशीर फी पाठवा, रक्कम सुटेल.",
        ],
        "account_suspension": [
            "संशयास्पद हालचालीमुळे {hours} तासांत खाते निलंबित होईल. आत्ता {link} वर तपासा.",
            "तातडीचे: अनोळखी लॉगिन आढळले. {hours} तासांत पासवर्ड कन्फर्म करा.",
        ],
    },
    "pa": {
        "otp_request": [
            "{bank}: {amount} ਦੀ ਪੇਮੇਂਟ ਲਈ OTP {otp} ਦੀ ਬੇਨਤੀ। ਤੁਸੀਂ ਨਹੀਂ ਕੀਤਾ ਤਾਂ ਫੌਰਨ ਕਾਲ ਕਰੋ।",
            "OTP {otp} ਸਾਂਝਾ ਕਰੋ ਅਤੇ {app} ਪੁਸ਼ਟੀ ਪੂਰੀ ਕਰੋ।",
        ],
        "payment_pressure": [
            "ਆਖਰੀ ਚੇਤਾਵਨੀ: ਕਾਰਡ 'ਤੇ {amount} ਬਾਕੀ। ਅੱਜ {link} 'ਤੇ ਭਰੋ ਨਹੀਂ ਤਾਂ ਕੇਸ ਹੋਵੇਗਾ।",
            "ਕਰਜ਼ੇ ਦੀ ਕਿਸਤ {amount} ਬਾਕੀ ਹੈ। ਅੱਜ ਭਰੋ ਨਹੀਂ ਤਾਂ ਏਜੰਟ ਘਰ ਆਵੇਗਾ।",
        ],
        "investment_crypto": [
            "ਗਾਰੰਟੀ ਸ਼ੁਦਾ ਮੁਨਾਫ਼ਾ: {amount} ਨੂੰ 30 ਦਿਨਾਂ ਵਿੱਚ 10x ਕਰੋ। ਅੱਜ {link} 'ਤੇ ਜੁੜੋ।",
            "ਕ੍ਰਿਪਟੋ ਸਿਗਨਲ - ਆਖਰੀ ਮੌਕਾ। {amount} ਭਰੋ, ਪੈਸਾ ਦੁੱਗਣਾ ਹੋਵੇਗਾ।",
        ],
        "lottery_prize": [
            "ਸ਼ਾਬਾਸ਼! ਤੁਸੀਂ {amount} ਜਿੱਤੇ ਹੋ। {hours} ਘੰਟਿਆਂ ਵਿੱਚ {link} 'ਤੇ ਕਲੇਮ ਕਰੋ।",
            "ਤੁਹਾਡਾ ਨੰਬਰ {amount} ਇਨਾਮ ਲਈ ਚੁਣਿਆ ਗਿਆ ਹੈ। ਹੁਣੇ ਏਜੰਟ ਕੋਲ ਕਲੇਮ ਕਰੋ।",
        ],
        "job_task": [
            "ਘਰ ਬੈਠੇ ਰੋਜ਼ {amount} ਕਮਾਓ। {link} 'ਤੇ ਰਜਿਸਟਰ ਕਰੋ, ਸਿਰਫ਼ {amount} ਫੀਸ।",
            "ਸ਼ਾਬਾਸ਼, ਤੁਸੀਂ ਵਰਕ-ਫਰਾਮ-ਹੋਮ ਨੌਕਰੀ ਲਈ ਚੁਣੇ ਗਏ ਹੋ। {amount} ਸੁਰੱਖਿਆ ਫੀਸ ਭੇਜੋ।",
        ],
        "gov_bank_impersonation": [
            "{bank} ਚੇਤਾਵਨੀ: KYC ਖਤਮ ਹੈ। ਹੁਣੇ {link} 'ਤੇ ਅੱਪਡੇਟ ਕਰੋ ਨਹੀਂ ਤਾਂ ਖਾਤਾ ਫ੍ਰੀਜ਼।",
            "RBI ਨੋਟਿਸ: ਅੱਜ {link} 'ਤੇ ਬੈਂਕ ਵੇਰਵੇ ਤਸਦੀਕ ਕਰੋ ਨਹੀਂ ਤਾਂ ਪੈਸੇ ਬਲਾਕ।",
        ],
        "delivery_scam": [
            "ਪਾਰਸਲ ਕਸਟਮਜ਼ ਵਿੱਚ ਰੁਕਿਆ ਹੈ। {amount} ਫੀਸ {link} 'ਤੇ ਭਰੋ।",
            "ਡਿਲੀਵਰੀ ਅਸਫਲ - ਪਤਾ ਅਧੂਰਾ। {amount} ਫੀਸ ਦੇ ਕੇ {link} 'ਤੇ ਦੁਬਾਰਾ ਸਮਾਂ ਬਣਾਓ।",
        ],
        "advance_fee": [
            "ਦੋਸਤ, ਮੇਰੇ ਕੋਲ {amount} ਦਾ ਡੱਬਾ ਹੈ। ਟ੍ਰਾਂਸਫਰ ਫੀਸ {amount} ਭੇਜੋ।",
            "ਤੁਹਾਨੂੰ {amount} ਦੀ ਵਿਰਾਸਤ ਮਿਲੀ ਹੈ। ਕਾਨੂੰਨੀ ਫੀਸ ਭੇਜੋ, ਰਕਮ ਜਾਰੀ ਹੋਵੇਗੀ।",
        ],
        "account_suspension": [
            "ਸ਼ੱਕੀ ਗਤੀਵਿਧੀ ਕਾਰਨ {hours} ਘੰਟਿਆਂ ਵਿੱਚ ਖਾਤਾ ਮੁਅੱਤਲ ਹੋਵੇਗਾ। ਹੁਣੇ {link} 'ਤੇ ਤਸਦੀਕ ਕਰੋ।",
            "ਤੁਰੰਤ: ਅਣਜਾਣ ਲੌਗਿਨ ਮਿਲਿਆ। {hours} ਘੰਟਿਆਂ ਵਿੱਚ ਪਾਸਵਰਡ ਪੁਸ਼ਟੀ ਕਰੋ।",
        ],
    },
    "gu": {
        "otp_request": [
            "{bank}: {amount} માટે OTP {otp} ની વિનંતી. તમે કરેલ ન હોય તો તરત કૉલ કરો.",
            "OTP {otp} શેર કરો અને {app} ચકાસણી પૂર્ણ કરો.",
        ],
        "payment_pressure": [
            "છેલ્લી ચેતવણી: કાર્ડ પર {amount} બાકી. આજે {link} પર ભરો નહીં તો કેસ થશે.",
            "લોનની કિસ્ત {amount} બાકી છે. આજે ભરો નહીં તો એજન્ટ ઘરે આવશે.",
        ],
        "investment_crypto": [
            "ગેરંટીડ રિટર્ન: {amount} ને 30 દિવસમાં 10x કરો. આજે {link} પર જોડાઓ.",
            "ક્રિપ્ટો સિગ્નલ - છેલ્લી તક. {amount} જમા કરો, પૈસા બમણા થશે.",
        ],
        "lottery_prize": [
            "અભિનંદન! તમે {amount} જીત્યા છો. {hours} કલાકમાં {link} પર ક્લેઇમ કરો.",
            "તમારો નંબર {amount} ઇનામ માટે પસંદ થયો છે. હમણાં એજન્ટ પાસે ક્લેઇમ કરો.",
        ],
        "job_task": [
            "ઘરે બેઠા રોજ {amount} કમાઓ. {link} પર રજીસ્ટર કરો, ફક્ત {amount} ફી.",
            "અભિનંદન, તમે વર્ક-ફ્રોમ-હોમ નોકરી માટે પસંદ થયા છો. {amount} સુરક્ષા ફી મોકલો.",
        ],
        "gov_bank_impersonation": [
            "{bank} એલર્ટ: KYC સમાપ્ત થઈ છે. તરત {link} પર અપડેટ કરો નહીં તો ખાતું ફ્રીઝ.",
            "RBI નોટિસ: આજે {link} પર બેંક વિગતો ચકાસો નહીં તો ફંડ બ્લોક.",
        ],
        "delivery_scam": [
            "પાર્સલ કસ્ટમ્સમાં અટક્યું છે. {amount} ફી {link} પર ભરો.",
            "ડિલિવરી નિષ્ફળ - સરનામું અધૂરું. {amount} ફી આપીને {link} પર ફરી સમય નક્કી કરો.",
        ],
        "advance_fee": [
            "મિત્ર, મારી પાસે {amount} નો ડબ્બો છે. ટ્રાન્સફર ફી {amount} મોકલો.",
            "તમને {amount} ની વારસાની રકમ મળી છે. કાનૂની ફી મોકલો, રકમ છૂટી થશે.",
        ],
        "account_suspension": [
            "શંકાસ્પદ પ્રવૃત્તિથી {hours} કલાકમાં ખાતું સસ્પેન્ડ થશે. હમણાં {link} પર ચકાસો.",
            "તાત્કાલિક: અજાણ્યો લોગિન મળ્યો. {hours} કલાકમાં પાસવર્ડ કન્ફર્મ કરો.",
        ],
    },
    "ta": {
        "otp_request": [
            "{bank}: {amount} செலுத்துதலுக்கான OTP {otp}. நீங்கள் அனுப்பவில்லை என்றால் உடனே அழையுங்கள்.",
            "OTP {otp} ஐப் பகிர்ந்து {app} சரிபார்ப்பை முடிக்கவும்.",
        ],
        "payment_pressure": [
            "இறுதி எச்சரிக்கை: கார்டில் {amount} நிலுவை. இன்றே {link} இல் செலுத்துங்கள்.",
            "கடன் தவணை {amount} நிலுவையில் உள்ளது. இன்றே செலுத்துங்கள்.",
        ],
        "investment_crypto": [
            "உறுதியான லாபம்: {amount} ஐ 30 நாட்களில் 10x ஆக்குங்கள். {link} இல் சேருங்கள்.",
            "கிரிப்டோ சிக்னல் - கடைசி வாய்ப்பு. {amount} செலுத்துங்கள், பணம் இரட்டிப்பாகும்.",
        ],
        "lottery_prize": [
            "வாழ்த்துகள்! நீங்கள் {amount} வென்றீர்கள். {hours} மணி நேரத்தில் {link} இல் கோருங்கள்.",
            "உங்கள் எண் {amount} பரிசுக்கு தேர்ந்தெடுக்கப்பட்டது. இப்போதே கோருங்கள்.",
        ],
        "job_task": [
            "வீட்டில் இருந்து தினமும் {amount} சம்பாதிக்கலாம். {link} இல் பதிவு செய்யுங்கள்.",
            "வாழ்த்துகள், வொர்க்-ஃப்ரம்-ஹோம் வேலைக்கு தேர்ந்தெடுக்கப்பட்டீர்கள். {amount} கட்டணம் அனுப்புங்கள்.",
        ],
        "gov_bank_impersonation": [
            "{bank} எச்சரிக்கை: KYC காலாவதி. உடனே {link} இல் புதுப்பிக்கவும்.",
            "RBI அறிவிப்பு: இன்று {link} இல் வங்கி விவரங்களை சரிபார்க்கவும்.",
        ],
        "delivery_scam": [
            "பொதி சுங்கத்தில் தடுக்கப்பட்டுள்ளது. {amount} கட்டணம் {link} இல் செலுத்துங்கள்.",
            "டெலிவரி தோல்வி - முகவரி முழுமையற்றது. {link} இல் மீண்டும் நேரம் நிச்சயிக்கவும்.",
        ],
        "advance_fee": [
            "நண்பரே, என்னிடம் {amount} பெட்டி உள்ளது. பரிமாற்றக் கட்டணம் {amount} அனுப்புங்கள்.",
            "உங்களுக்கு {amount} பரம்பரை சொத்து கிடைத்துள்ளது. சட்டக் கட்டணம் அனுப்புங்கள்.",
        ],
        "account_suspension": [
            "சந்தேகத்திற்குரிய செயல்பாடு காரணமாக {hours} மணி நேரத்தில் கணக்கு இடைநீக்கம். {link} இல் சரிபார்க்கவும்.",
            "அவசரம்: அறியப்படாத உள்நுழைவு. {hours} மணி நேரத்தில் கடவுச்சொல்லை உறுதிப்படுத்துங்கள்.",
        ],
    },
    "te": {
        "otp_request": [
            "{bank}: {amount} చెల్లింపు కోసం OTP {otp}. మీరు చేయలేదు అంటే వెంటనే కాల్ చేయండి.",
            "OTP {otp} పంచి {app} ధృవీకరణ పూర్తి చేయండి.",
        ],
        "payment_pressure": [
            "చివరి హెచ్చరిక: కార్డుపై {amount} బకాయి. ఈరోజే {link} లో చెల్లించండి.",
            "రుణ వాయిదా {amount} బకాయి. ఈరోజే చెల్లించండి లేదంటే ఏజెంట్ ఇంటికి వస్తారు.",
        ],
        "investment_crypto": [
            "హామీ రాబడి: {amount} ను 30 రోజుల్లో 10x చేయండి. ఈరోజే {link} లో చేరండి.",
            "క్రిప్టో సిగ్నల్ - చివరి అవకాశం. {amount} జమ చేయండి, డబ్బు రెట్టింపు అవుతుంది.",
        ],
        "lottery_prize": [
            "అభినందనలు! మీరు {amount} గెలిచారు. {hours} గంటల్లో {link} లో క్లెయిమ్ చేయండి.",
            "మీ నంబర్ {amount} బహుమతికి ఎంపికైంది. వెంటనే క్లెయిమ్ చేయండి.",
        ],
        "job_task": [
            "ఇంట్లో ఉండి రోజూ {amount} సంపాదించండి. {link} లో నమోదు చేయండి.",
            "అభినందనలు, వర్క్-ఫ్రమ్-హోమ్ ఉద్యోగానికి ఎంపికయ్యారు. {amount} సెక్యూరిటీ ఫీజు పంపండి.",
        ],
        "gov_bank_impersonation": [
            "{bank} హెచ్చరిక: KYC గడువు ముగిసింది. వెంటనే {link} లో అప్డేట్ చేయండి.",
            "RBI నోటీసు: ఈరోజే {link} లో బ్యాంక్ వివరాలు ధృవీకరించండి.",
        ],
        "delivery_scam": [
            "పార్సెల్ కస్టమ్స్‌లో ఆగింది. {amount} ఫీజు {link} లో చెల్లించండి.",
            "డెలివరీ విఫలం - చిరునామా పూర్తి కాలేదు. {link} లో మళ్లీ సమయం నిర్ణయించండి.",
        ],
        "advance_fee": [
            "స్నేహితుడా, నా దగ్గర {amount} బాక్స్ ఉంది. బదిలీ ఫీజు {amount} పంపండి.",
            "మీకు {amount} వారసత్వ ఆస్తి వచ్చింది. చట్టపరమైన ఫీజు పంపండి.",
        ],
        "account_suspension": [
            "అనుమానాస్పద కార్యకలాపాల కారణంగా {hours} గంటల్లో ఖాతా సస్పెన్షన్. {link} లో ధృవీకరించండి.",
            "అత్యవసరం: తెలియని లాగిన్ గుర్తించబడింది. {hours} గంటల్లో పాస్‌వర్డ్ నిర్ధారించండి.",
        ],
    },
    "es": {
        "otp_request": [
            "{bank}: se solicito el OTP {otp} para un pago de {amount}. Si no fuiste tu, llama ya.",
            "Comparte el codigo {otp} para verificar tu {app}. El soporte nunca pide el codigo.",
        ],
        "payment_pressure": [
            "AVISO FINAL: {amount} pendientes en tu tarjeta. Paga hoy en {link} o inicia demanda en {hours} horas.",
            "Tu prestamo de {amount} esta vencido. Paga hoy o los agentes visitaran tu domicilio.",
        ],
        "investment_crypto": [
            "Ganancias garantizadas: convierte {amount} en 10x en 30 dias. Unete en {link}",
            "Senal cripto insider - ultima oportunidad. Deposita {amount} hoy y duplica.",
        ],
        "lottery_prize": [
            "FELICIDADES! Ganaste {amount} en la loteria internacional. Reclama en {link} en {hours} horas.",
            "Tu numero fue seleccionado para un premio de {amount}. Reclama con el agente ahora.",
        ],
        "job_task": [
            "Gana {amount} al dia desde casa. Registrate en {link} - solo {amount} de inscripcion.",
            "Felicidades, fuiste seleccionado para un trabajo remoto. Paga {amount} de garantia para empezar.",
        ],
        "gov_bank_impersonation": [
            "ALERTA {bank}: Tu KYC expiro. Actualizalo ya en {link} o se congela la cuenta.",
            "Aviso oficial: verifica tus datos bancarios hoy en {link} para mantener tus fondos.",
        ],
        "delivery_scam": [
            "Tu paquete esta retenido en aduana. Paga {amount} en {link} para liberar la entrega.",
            "Entrega fallida - direccion incompleta. Reprograma pagando {amount} en {link}",
        ],
        "advance_fee": [
            "Amigo, tengo un paquete de {amount}. Paga la tarifa de transferencia de {amount} para recibirlo.",
            "Heredaste {amount} de un cliente extranjero. Paga los gastos legales para liberar los fondos.",
        ],
        "account_suspension": [
            "Tu cuenta se suspende en {hours} horas por actividad sospechosa. Verifica en {link} ahora.",
            "URGENTE: login sospechoso detectado. Confirma tu contrasena en {hours} horas o pierdes acceso.",
        ],
    },
    "pt": {
        "otp_request": [
            "{bank}: OTP {otp} solicitado para pagamento de {amount}. Se nao foi voce, ligue agora.",
            "Compartilhe o codigo {otp} para verificar seu {app}. O suporte nunca pede o codigo.",
        ],
        "payment_pressure": [
            "AVISO FINAL: {amount} pendentes no seu cartao. Pague hoje em {link} ou acao judicial em {hours} horas.",
            "Seu emprestimo de {amount} esta vencido. Quite hoje ou agentes visitarao seu endereco.",
        ],
        "investment_crypto": [
            "Retorno garantido: transforme {amount} em 10x em 30 dias. Entre em {link}",
            "Sinal cripto insider - ultima chance. Deposite {amount} hoje e duplique.",
        ],
        "lottery_prize": [
            "PARABENS! Voce ganhou {amount} na loteria internacional. Resgate em {link} em {hours} horas.",
            "Seu numero foi sorteado para um premio de {amount}. Resgate com o agente agora.",
        ],
        "job_task": [
            "Ganhe {amount} por dia em casa. Cadastre-se em {link} - apenas {amount} de taxa.",
            "Parabens, voce foi selecionado para um trabalho remoto. Pague {amount} de garantia para comecar.",
        ],
        "gov_bank_impersonation": [
            "ALERTA {bank}: Seu KYC expirou. Atualize em {link} ou a conta sera bloqueada.",
            "Aviso oficial: verifique seus dados bancarios hoje em {link} para manter os fundos.",
        ],
        "delivery_scam": [
            "Seu pacote esta retido na alfandega. Pague {amount} em {link} para liberar a entrega.",
            "Entrega falhou - endereco incompleto. Reprogarme pagando {amount} em {link}",
        ],
        "advance_fee": [
            "Amigo, tenho um cofre de {amount}. Pague a taxa de transferencia de {amount} para receber.",
            "Voce herdou {amount} de um cliente estrangeiro. Pague as taxas legais para liberar.",
        ],
        "account_suspension": [
            "Sua conta sera suspensa em {hours} horas por atividade suspeita. Verifique em {link} agora.",
            "URGENTE: login suspeito detectado. Confirme sua senha em {hours} horas ou perde acesso.",
        ],
    },
    "fr": {
        "otp_request": [
            "{bank} : OTP {otp} demande pour un paiement de {amount}. Si ce n'etait pas vous, appelez vite.",
            "Partagez le code {otp} pour verifier votre {app}. Le support ne demande jamais le code.",
        ],
        "payment_pressure": [
            "DERNIER AVERTISSEMENT : {amount} dus sur votre carte. Payez aujourd'hui sur {link} sinon poursuite.",
            "Votre pret de {amount} est echu. Reglez aujourd'hui ou un agent viendra chez vous.",
        ],
        "investment_crypto": [
            "Rendement garanti : transformez {amount} en 10x en 30 jours. Rejoignez {link}",
            "Signal crypto insider - derniere chance. Deposez {amount} aujourd'hui et doublez.",
        ],
        "lottery_prize": [
            "FELICITATIONS ! Vous avez gagne {amount} a la loterie internationale. Reclamez sur {link}",
            "Votre numero a ete tire pour un prix de {amount}. Reclamez aupres de l'agent maintenant.",
        ],
        "job_task": [
            "Gagnez {amount} par jour depuis chez vous. Inscrivez-vous sur {link} - seulement {amount}.",
            "Felicitations, vous etes selectionne pour un emploi a domicile. Payez {amount} de caution.",
        ],
        "gov_bank_impersonation": [
            "ALERTE {bank} : Votre KYC a expire. Mettez-le a jour sur {link} ou le compte sera gele.",
            "Avis officiel : verifiez vos donnees bancaires aujourd'hui sur {link}.",
        ],
        "delivery_scam": [
            "Votre colis est retenu en douane. Payez {amount} sur {link} pour la livraison.",
            "Livraison echouee - adresse incomplete. Reprogrammez en payant {amount} sur {link}",
        ],
        "advance_fee": [
            "Cher ami, je dispose d'un coffre de {amount}. Payez les frais de transfert de {amount}.",
            "Vous avez herite de {amount}. Payez les frais juridiques pour de Bloquer les fonds.",
        ],
        "account_suspension": [
            "Votre compte sera suspendu dans {hours} heures pour activite suspecte. Verifiez sur {link}.",
            "URGENT : connexion suspecte detectee. Confirmez votre mot de passe sous {hours} heures.",
        ],
    },
    "ar": {
        "otp_request": [
            "{bank}: تم طلب رمز التحقق {otp} لدفع {amount}. إذا لم تكن أنت فاتصل فوراً.",
            "شارك رمز التحقق {otp} لإكمال تأكيد {app}. الدعم لا يطلب الرمز أبداً.",
        ],
        "payment_pressure": [
            "تحذير أخير: مبلغ {amount} مستحق على بطاقتك. ادفع اليوم عبر {link} وإلا بدأت الإجراءات.",
            "قسط القرض {amount} متأخر. سدده اليوم وإلا جاء المندوب إلى منزلك.",
        ],
        "investment_crypto": [
            "أرباح مضمونة: حوّل {amount} إلى 10 أضعاف في 30 يوماً. انضم عبر {link}",
            "إشارة عملات رقمية - فرصة أخيرة. أودع {amount} اليوم وضاعف أموالك.",
        ],
        "lottery_prize": [
            "مبروك! لقد ربحت {amount} في اليانصيب العالمي. اطلبها خلال {hours} ساعة عبر {link}",
            "تم اختيار رقمك لجائزة {amount}. اطلبها من الوكيل الآن.",
        ],
        "job_task": [
            "اربح {amount} يومياً من المنزل. سجل عبر {link} - رسوم {amount} فقط.",
            "مبروك، تم اختيارك للعمل من المنزل. ادفع {amount} تأمين للبدء.",
        ],
        "gov_bank_impersonation": [
            "تنبيه {bank}: انتهت صلاحية KYC. حدثها الآن عبر {link} وإلا جُمد الحساب.",
            "إشعار رسمي: تحقق من بياناتك المصرفية اليوم عبر {link}",
        ],
        "delivery_scam": [
            "طرودك محتجز في الجمارك. ادفع {amount} عبر {link} لتحرير التسليم.",
            "فشل التسليم - العنوان ناقص. أعد الجدولة بدفع {amount} عبر {link}",
        ],
        "advance_fee": [
            "صديقي، لدي صندوق بقيمة {amount}. ادفع رسوم التحويل {amount} لاستلامه.",
            "ورثت {amount} من عميل أجنبي. ادفع الرسوم القانونية لتحرير الأموال.",
        ],
        "account_suspension": [
            "سيتم تعليق حسابك خلال {hours} ساعة بسبب نشاط مشبوه. تحقق عبر {link} الآن.",
            "عاجل: تم رصد دخول مشبوه. أكد كلمة المرور خلال {hours} ساعة أو تفقد الوصول.",
        ],
    },
    "id": {
        "otp_request": [
            "{bank}: OTP {otp} diminta untuk pembayaran {amount}. Jika bukan Anda, segera hubungi kami.",
            "Bagikan OTP {otp} untuk verifikasi {app}. Petugas tidak pernah meminta OTP.",
        ],
        "payment_pressure": [
            "PERINGATAN TERAKHIR: {amount} tertunggak di kartu Anda. Bayar hari ini di {link} atau dituntut.",
            "Cicilan pinjaman {amount} jatuh tempo. Lunasi hari ini atau agen datang ke rumah.",
        ],
        "investment_crypto": [
            "Untung dijamin: ubah {amount} menjadi 10x dalam 30 hari. Gabung di {link}",
            "Sinyal kripto insider - kesempatan terakhir. Setor {amount} hari ini, uang berlipat.",
        ],
        "lottery_prize": [
            "SELAMAT! Anda memenangkan {amount} dalam lotere internasional. Klaim di {link} dalam {hours} jam.",
            "Nomor Anda terpilih untuk hadiah {amount}. Klaim ke agen sekarang.",
        ],
        "job_task": [
            "Hasilkan {amount} per hari dari rumah. Daftar di {link} - hanya biaya {amount}.",
            "Selamat, Anda terpilih untuk kerja dari rumah. Bayar {amount} jaminan untuk mulai.",
        ],
        "gov_bank_impersonation": [
            "PERINGATAN {bank}: KYC Anda kedaluwarsa. Perbarui di {link} atau rekening dibekukan.",
            "Pemberitahuan resmi: verifikasi data bank hari ini di {link}",
        ],
        "delivery_scam": [
            "Paket Anda tertahan di bea cukai. Bayar {amount} di {link} untuk melanjutkan pengiriman.",
            "Pengiriman gagal - alamat tidak lengkap. Jadwalkan ulang dengan bayar {amount} di {link}",
        ],
        "advance_fee": [
            "Sahabat, saya punya peti berisi {amount}. Bayar biaya transfer {amount} untuk menerimanya.",
            "Anda menerima warisan {amount}. Bayar biaya hukum agar dana dilepaskan.",
        ],
        "account_suspension": [
            "Rekening Anda diblokir dalam {hours} jam karena aktivitas mencurigakan. Verifikasi di {link} sekarang.",
            "URGENT: login mencurigakan terdeteksi. Konfirmasi kata sandi dalam {hours} jam.",
        ],
    },
}

_SLOTS = {
    "otp": lambda rng: f"{rng.randint(1000, 9999)}",
    "code": lambda rng: f"{rng.randint(10000, 99999)}",
    "amount": lambda rng: f"{rng.choice([499, 999, 1500, 4999, 9999, 25000, 49999, 78000])}",
    "hours": lambda rng: f"{rng.choice([12, 24, 48, 72])}",
    "bank": lambda rng: rng.choice(BANKS),
    "app": lambda rng: rng.choice(APPS),
    "link": lambda rng: rng.choice(LINKS),
}


def _fill(template: str, rng: random.Random) -> str:
    out = template
    for slot, fn in _SLOTS.items():
        placeholder = "{" + slot + "}"
        while placeholder in out:
            out = out.replace(placeholder, fn(rng), 1)
    return out


def generate_lang_category(lang: str, category: str, n: int, rng: random.Random) -> list[str]:
    templates = TEMPLATES.get(lang, {}).get(category, [])
    if not templates:
        return []
    texts = []
    for _ in range(n):
        texts.append(_fill(rng.choice(templates), rng))
    return texts


def generate_synthetic(cap_per_lang_category: int = 60, seed: int = 42) -> list[dict]:
    """Deterministic synthetic corpus: scams per (lang, category) + hard negatives.

    cap_per_lang_category bounds the synthetic share so real data keeps
    dominating: 14 langs x 9 categories x 60 = ~7.5k scam rows max, plus
    ~4-6k hard negatives. Tuned so synthetic stays well under 10% of total.
    """
    rng = random.Random(seed)
    records: list[dict] = []

    for lang in LANGS:
        for category in CATEGORIES:
            for text in generate_lang_category(lang, category, cap_per_lang_category, rng):
                records.append(
                    make_message(text, [category], f"{SOURCE_PREFIX}_{lang}", language=lang)
                )
        negatives = HARD_NEGATIVES.get(lang, [])
        for i in range(max(len(negatives) * 12, 60)):
            text = _fill(rng.choice(negatives), rng)
            records.append(make_message(text, [], f"{SOURCE_PREFIX}_{lang}", language=lang))

    return records
