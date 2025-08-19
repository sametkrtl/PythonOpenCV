# ------------------------------------------------------------------
#  TuM LISTELERIN BIRLEsTIRILIP DOgRULANDIgI GuNCEL KOD BLOgU
# ------------------------------------------------------------------

# --- 1. DuZ (GENEL) LISTELER ---
# Bu listeler, hem uzunluk bazli gruplama hem de genel kullanim icin temel veri kaynagidir.

sirket_samples = [
    "BIM", "sOK", "A101", "IKEA", "Metro", "Migros", "Real", "Kipa", "Sarar", "Damat", "Koton", "Zara", "Mango", "Puma", "Nike", "Bosch", "Beko", "Arcelik", "Vestel", "Profilo", "Siemens", "Samsung", "LG", "Philips", "Koctas", "Teknosa", "Pegasus", "Akbank", "ETI", "ulker", "Nestle", "Pepsi", "Boyner", "Kigili", "H&M", "Vans", "Adidas", "Reebok", "Network", "Beymen", "Yapi Kredi", "VakifBank", "Halkbank", "ING Bank", "DenizBank", "Unilever", "P&G", "Colin's", "DeFacto", "Bauhaus", "Turkcell", "Vodafone", "Carrefour", "Intersport", "Decathlon", "Is Bankasi", "LC Waikiki", "MediaMarkt", "Converse", "Timberland", "Columbia", "CarrefourSA", "Altinyildiz", "Leroy Merlin", "Ziraat Bankasi", "Turkish Airlines", "Coca Cola", "QNB Finansbank", "The North Face", "Turk Telekom"
]

ad_soyad_samples = [
    "Ali", "Can", "Ahmet", "Zehra", "Mehmet", "Fatma", "Seda Ates", "Gul Akgul", "Cem Balci", "Sibel Tuna", "Burak Kose", "Tolga ozgur", "Murat Isik", "ozlem Oral", "Deniz Tekin", "Burcu cag", "Ayse Kaya", "Emine Tas", "Hatice Uz", "Zeliha Can", "Fadime oz", "Serkan Peker", "Dilek Soylu", "Emre Basak", "Serap Gur", "Ercan Sever", "Melike Karan", "Volkan cam", "sule Nas", "Kadir Sonmez", "Ahmet Kaya", "Hacer Bulut", "Mustafa sahin", "Emine Yildiz", "Ali ozkan", "Ibrahim Dogan", "Zeynep Kilic", "Huseyin Aslan", "Zeliha Polat", "Osman Akin", "Pinar Durak", "Ahmet Yilmaz", "Mehmet Kaya", "Fatma Demir", "Ayse celik", "Hatice Arslan", "Suleyman Yucel", "Hacer Turk", "Meryem ozturk", "Yasar Tunc", "Rukiye Aydin", "Recep ozdemir", "Hanife Basaran", "Yusuf Koc", "Gulay Karaca", "Hasan Ucar", "Nuriye ciftci", "Kemal Duman", "Necla Keskin", "Ramazan Bulut", "Melahat Tas", "Bayram Yavuz", "Saliha Ceylan", "omer Sezer", "Cemile Ergin", "Halil Kurt", "Fatih Uzun", "Mehmet Demir", "Ibrahim ozkan", "Abdullah Gunes", "Abdullah Dogan", "Mustafa Yilmaz", "Suleyman Erdogan"
]

para_samples = [
    "50 TL", "75 TL", "80 TL", "120 TL", "300 TL", "480 TL", "650 TL", "899 TL", "950 TL", "150 EUR", "200 USD", "1.250 TL", "1.500 TL", "1.800 TL", "2.100 TL", "2.350 TL", "2.900 TL", "3.750 TL", "4.500 TL", "5.500 TL", "6.500 TL", "7.250 TL", "8.900 TL", "9.750 TL", "1.200 EUR", "1.500 USD", "2.250 USD", "2.800 EUR", "6.250 TL", "8.500 EUR", "12.750 TL", "15.750 TL", "19.750 TL", "22.500 TL", "25.000 TL", "28.500 TL", "38.000 TL", "45.000 TL", "55.000 TL", "95.000 TL", "10.000 USD", "12.500 EUR", "15.500 USD", "18.000 EUR", "25.000 USD", "65.000 EUR", "75.000 USD", "100.000 TL", "125.000 TL", "140.000 TL", "175.000 TL", "180.000 TL", "220.000 TL", "350.000 TL"
]

adres_samples = [
    "Ist", "Ank", "Bursa", "Izmir", "Adana", "Ankara", "Istanbul", "Antalya", "Denizli", "Eskisehir", "Samsun", "Gaziantep", "Atakum", "Kocasinan", "Konak/Izm", "No:15 Kadikoy", "Cad. cankaya", "Bursa/Mrkz", "Kadikoy/Ist", "cankaya/Ank", "Mah. Inonu Cad", "Zafer Mah. Kordon", "Ataturk Mah. Cad", "Sok. No:23 Bursa", "Barbaros Mah. Ataturk", "Gazi Bulvari No:42", "Fatih Mah. Sinan S.", "Kultur Mah. Hurriyet", "Yildirim Mah. Istiklal", "Guzelyali Mah. sehit C.", "Kocatepe Mah. Mithatp.", "Fevzi Pasa Mah. Gaziler", "Mehmet Akif Mah. can.", "Yenisehir Mah. Cumhur. B.", "Selcuklu Mah. Mevlana Bul.", "Inonu Mah. Ataturk Cad. ", "camlik Mah. Sahil Yolu No:", "Yesilova Mah. Ankara Cad. ", "Karsiyaka Mah. Izmir Yolu N", "Bati Mah. Londra Asfalti No", "Dogu Mah. Bagdat Cad. No:2", "Merkez Mah. Sakarya Cad. No", "Guney Mah. Antalya Bulvari ", "Kuzey Mah. Sivas Yolu No:87", "Anadolu Mah. Trabzon Cad. N", "Rumeli Mah. Edirne Yolu No:", "Orta Mah. Konya Cad. No:54 ", "Ic Mah. Ankara Bulvari No:76", "Dis Mah. Istanbul Yolu No:1", "Alt Mah. Bursa Cad. No:43 Os", "ust Mah. Adana Yolu No:85 S", "on Mah. Mersin Bulvari No:16", "Arka Mah. Gaziantep Cad. No", "Orman Mah. Tabiat Yolu No:18", "Deniz Mah. Sahil Cad. No:7 K", "Dag Mah. Uludag Sok. No:31 O", "Vadi Mah. camlik Cad. No:64 ", "Tepe Mah. Yuksek Sok. No:19 c", "Duz Mah. Ovacik Yolu No:145 ", "Egik Mah. Yamac Cad. No:52 Ka", "Genis Mah. Acik Alan Sok. No:", "Dar Mah. Kisa Yol No:6 Kadiko", "Uzun Mah. Mesafe Cad. No:174 ", "Kisa Mah. Yakin Sok. No:11 Ko", "Buyuk Mah. Genis Bulvari No:2", "Kucuk Mah. Minik Sok. No:4 Be", "Eski Mah. Tarih Cad. No:99 Ulus/Ankara", "Zafer Mah. Kordon Cad. No:8 Konak/Izmir", "Yeni Mah. Modern Bulvari No:156 Bornova/Izmir", "Ataturk Mah. Inonu Cad. No:15 Kadikoy/Istanbul", "Fatih Mah. Mimar Sinan Sok. No:23 Osmangazi/Bursa", "Cumhuriyet Mah. Gazi Bulvari No:42 cankaya/Ankara", "Kultur Mah. Hurriyet Cad. No:12 Seyhan/Adana", "Yildirim Mah. Istiklal Sok. No:5 Merkez/Eskisehir", "Guzelyali Mah. sehit Cad. No:34 Atakum/Samsun", "Barbaros Mah. Ataturk Bulvari No:67 Muratpasa/Antalya", "Kocatepe Mah. Mithatpasa Cad. No:56 Kocasinan/Kayseri", "Mehmet Akif Mah. cankiri Cad. No:89 Altindag/Ankara", "Fevzi Pasa Mah. Gaziler Cad. No:91 sahinbey/Gaziantep", "Yenisehir Mah. Cumhuriyet Bulvari No:78 Pamukkale/Denizli", "Selcuklu Mah. Mevlana Bulvari No:25 Selcuklu/Konya", "Inonu Mah. Ataturk Cad. No:47 Tepebasi/Eskisehir", "camlik Mah. Sahil Yolu No:13 Bodrum/Mugla", "Yesilova Mah. Ankara Cad. No:62 Osmangazi/Bursa", "Karsiyaka Mah. Izmir Yolu No:38 Torbali/Izmir", "Bati Mah. Londra Asfalti No:155 Esenyurt/Istanbul", "Dogu Mah. Bagdat Cad. No:244 Maltepe/Istanbul", "Merkez Mah. Sakarya Cad. No:71 Adapazari/Sakarya", "Guney Mah. Antalya Bulvari No:103 Kepez/Antalya", "Kuzey Mah. Sivas Yolu No:87 Melikgazi/Kayseri", "Anadolu Mah. Trabzon Cad. No:29 Atakum/Samsun", "Rumeli Mah. Edirne Yolu No:126 Arnavutkoy/Istanbul", "Orta Mah. Konya Cad. No:54 Selcuklu/Konya", "Ic Mah. Ankara Bulvari No:76 cankaya/Ankara", "Dis Mah. Istanbul Yolu No:198 Nilufer/Bursa", "Alt Mah. Bursa Cad. No:43 Osmangazi/Bursa", "ust Mah. Adana Yolu No:85 Seyhan/Adana", "on Mah. Mersin Bulvari No:167 Mezitli/Mersin", "Arka Mah. Gaziantep Cad. No:92 sehitkamil/Gaziantep", "Orman Mah. Tabiat Yolu No:18 Beykoz/Istanbul", "Deniz Mah. Sahil Cad. No:7 Kartal/Istanbul", "Dag Mah. Uludag Sok. No:31 Osmangazi/Bursa", "Vadi Mah. camlik Cad. No:64 Bornova/Izmir", "Tepe Mah. Yuksek Sok. No:19 cankaya/Ankara", "Duz Mah. Ovacik Yolu No:145 Tuzla/Istanbul", "Egik Mah. Yamac Cad. No:52 Karsiyaka/Izmir", "Genis Mah. Acik Alan Sok. No:28 Nilufer/Bursa", "Dar Mah. Kisa Yol No:6 Kadikoy/Istanbul", "Uzun Mah. Mesafe Cad. No:174 cankaya/Ankara", "Kisa Mah. Yakin Sok. No:11 Konak/Izmir", "Buyuk Mah. Genis Bulvari No:233 Osmangazi/Bursa", "Kucuk Mah. Minik Sok. No:4 Beyoglu/Istanbul", "Guzel Mah. Hos Sok. No:37 Nilufer/Bursa", "cirkin Mah. Kotu Yol No:83 Pendik/Istanbul", "Temiz Mah. Pak Cad. No:21 Kecioren/Ankara", "Kirli Mah. Pis Sok. No:77 Kartal/Istanbul", "Sessiz Mah. Sakin Yolu No:14 cankaya/Ankara", "Gurultulu Mah. samata Cad. No:118 Kadikoy/Istanbul", "Aydinlik Mah. Isik Bulvari No:65 Konak/Izmir", "Karanlik Mah. Golge Sok. No:39 Besiktas/Istanbul"
]

telefon_samples = [
    "05321234567", "05421234568", "05331234569", "05531234570", "05341234571", "05441234572", "05351234573", "05451234574", "05361234575", "05461234576", "05371234577", "05471234578", "05381234579", "05481234580", "05391234581", "05491234582", "05321111111", "05322222222", "05333333333", "05444444444", "05355555555", "05466666666", "05377777777", "05488888888", "05339999999", "05320000000", "05429876543", "05338765432", "05547654321", "05346543210", "05455432109", "05364321098", "05375210987", "05486109876", "05397098765", "05498987654", "05321357924", "05422468135", "05333691470", "05544702581", "05355813692", "05466924703", "05377035814", "05488146925", "05399258036", "05320369147", "05429470258", "05338581369", "05547692470", "05346703581", "05455814692", "05364925703", "05376036814", "05487147925", "05398259036", "05499360147"
]

tarih_samples = [
    "03.03.2023", "04.09.2023", "05.02.2023", "07.10.2022", "08.12.2022", "09.05.2023", "10.01.2024", "11.06.2023", "12.06.2022", "13.08.2023", "14.07.2024", "15.03.2024", "15.05.2023", "16.11.2022", "17.01.2024", "18.12.2023", "19.04.2024", "20.03.2022", "21.09.2022", "22.07.2023", "23.08.2024", "24.07.2023", "25.04.2024", "26.11.2022", "27.09.2022", "28.10.2023", "29.02.2024", "30.01.2024", "31.05.2024", "01.10.2022", "02.11.2024", "03.10.2022", "04.11.2023", "06.12.2022", "08.08.2023", "09.01.2024", "10.08.2022", "13.04.2024", "14.09.2022", "16.04.2024", "17.05.2022", "18.06.2023", "21.06.2023", "22.03.2024", "25.12.2023", "26.02.2024", "28.12.2023", "32.07.2023", "4 Eylul 2023", "8 Aralik 2022", "15 Mart 2024", "17 Ocak 2024", "22 Temmuz 2023", "31 Mayis 2024"
]

email_samples = [
    "ayse@outlook.com", "seda.ates@yahoo.com", "ahmet@gmail.com", "burak@outlook.com", "deniz.tekin@outlook.com", "serkan@hotmail.com", "emre@outlook.com", "tolga@hotmail.com", "ozlem@gmail.com", "sibel@yahoo.com", "volkan@hotmail.com", "burcu@gmail.com", "fatma.demir@yahoo.com", "zeynep@hotmail.com", "fadime@hotmail.com", "gulay@hotmail.com", "necla@hotmail.com", "melahat@outlook.com", "cemile@outlook.com", "serap.gur@gmail.com", "pinar.durak@yahoo.com", "cem.balci@hotmail.com", "sule.nas@yahoo.com", "mehmet.kaya@hotmail.com", "ali.ozkan@yahoo.com", "hatice@gmail.com", "hacer.turk@yahoo.com", "meryem@gmail.com", "yasar@gmail.com", "hanife@outlook.com", "nuriye@outlook.com", "ramazan@yahoo.com", "saliha@hotmail.com", "dilek.soylu@yahoo.com", "murat.isik@outlook.com", "ercan.sever@outlook.com", "melike.karan@gmail.com", "mustafa123@gmail.com", "emine.yildiz@hotmail.com", "ibrahim.dogan@outlook.com", "huseyin.aslan@gmail.com", "zeliha@yahoo.com", "osman.akin@outlook.com", "suleyman@gmail.com", "abdullah@outlook.com", "rukiye.aydin@hotmail.com", "recep.ozdemir@yahoo.com", "yusuf.koc@gmail.com", "hasan.ucar@yahoo.com", "kemal.duman@gmail.com", "bayram.yavuz@gmail.com", "omer.sezer@yahoo.com", "halil.kurt@gmail.com", "fatih.uzun@hotmail.com", "gul.akgul@gmail.com"
]

iban_samples = [
    "TR330006100519786457841326", "TR640004600119786543210987", "TR750001500658742039485760", "TR860010005018765432109876", "TR970006200119874563201234", "TR180009900658741250963847", "TR290001200519632540789513", "TR400010100119753951486270", "TR510006400658987654321098", "TR620004800519874563201357", "TR730001700119654783921046", "TR840009800658741852963047", "TR950006300519987654321579", "TR060010200119753846291570", "TR170001600658852741963048", "TR280009700519741963085274", "TR390006500119863529471604", "TR500004900658741963852074", "TR610001800519654387291046", "TR720010300119741852963074", "TR830006600658987321456078", "TR940009600519852741963047", "TR050001900119654789321046", "TR160010400658741963852074", "TR270006700519987654123578", "TR380004700119852963741046", "TR490002000658741963852074", "TR600009500519654789321046", "TR710006800119852741963074", "TR820010500658987654321078", "TR930001100519741963852074", "TR040009400119654789321046", "TR150006900658852741963074", "TR260002100519987321456078", "TR370004600119741963852074", "TR480010600658654789321046", "TR590009300519852741963074", "TR700007000119987654321078", "TR810001300658741963852074", "TR920009200519654789321046", "TR030010700119852741963074", "TR140007100658987321456078", "TR250002200519741963852074", "TR360004500119654789321046", "TR470009100658852741963074", "TR580007200119987321456078", "TR690010800519741963852074", "TR800001400658654789321046", "TR910009000119852741963074", "TR020007300658987321456078", "TR130002300519741963852074"
]

# --- 2. UZUNLUK BAZLI LISTELER (DOgRU FORMATTA VE DOgRU UZUNLUKLARLA) ---
# Bu listeler, kodunuzun beklentisine uygun olarak "sozluk listesi" formatindadir.

sirket_len_samples = [
    {'length': 3, 'samples': ['BIM', 'sOK', 'ETI']},
    {'length': 4, 'samples': ['A101', 'IKEA', 'Real', 'Kipa', 'Zara', 'Puma', 'Nike', 'Beko', 'Vans']},
    {'length': 5, 'samples': ['Metro', 'Sarar', 'Damat', 'Koton', 'Mango', 'Bosch', 'ulker', 'Pepsi', 'LG']},
    {'length': 6, 'samples': ['Migros', 'Kigili', 'H&M', 'Adidas', 'Reebok', 'Nestle', 'P&G', 'Arcelik', 'Vestel', 'Boyner']},
    {'length': 7, 'samples': ['Koctas', 'Teknosa', 'Pegasus', 'Akbank', 'Network', 'Beymen', 'Profilo', 'Siemens', 'Samsung', 'Philips', 'Unilever']},
    {'length': 8, 'samples': ['Turkcell', 'Vodafone', 'Colin’s', 'DeFacto', 'Bauhaus', 'VakifBank', 'Halkbank', 'ING Bank']},
    {'length': 9, 'samples': ['Carrefour', 'DenizBank', 'Yapi Kredi']},
    {'length': 10, 'samples': ['Is Bankasi', 'LC Waikiki', 'MediaMarkt', 'Converse', 'Coca Cola']},
    {'length': 11, 'samples': ['CarrefourSA', 'Altinyildiz', 'Timberland', 'Ziraat Bankasi']},
    {'length': 12, 'samples': ['Intersport', 'Decathlon', 'Turk Telekom']},
    {'length': 13, 'samples': ['Leroy Merlin', 'QNB Finansbank']},
    {'length': 14, 'samples': ['Turkish Airlines']},
    {'length': 15, 'samples': ['The North Face']}
]

ad_soyad_len_samples = [
    {'length': 3, 'samples': ['Ali', 'Can']},
    {'length': 4, 'samples': ['Ahmet', 'Zehra']},
    {'length': 5, 'samples': ['Mehmet', 'Fatma']},
    {'length': 6, 'samples': ['Emine Tas', 'Hatice Uz']},
    {'length': 7, 'samples': ['Zeliha Can', 'Fadime oz']},
    {'length': 8, 'samples': ['Seda Ates', 'Gul Akgul', 'Cem Balci', 'Sibel Tuna']},
    {'length': 9, 'samples': ['Ayse Kaya', 'Burak Kose', 'Tolga ozgur', 'Murat Isik', 'ozlem Oral']},
    {'length': 10, 'samples': ['Ahmet Kaya', 'Deniz Tekin', 'Burcu cag', 'Serkan Peker', 'Dilek Soylu', 'Emre Basak', 'Serap Gur']},
    {'length': 11, 'samples': ['Hacer Bulut', 'Ercan Sever', 'Melike Karan', 'Volkan cam', 'sule Nas', 'Kadir Sonmez']},
    {'length': 12, 'samples': ['Ahmet Yilmaz', 'Mehmet Kaya', 'Fatma Demir', 'Ayse celik', 'Pinar Durak']},
    {'length': 13, 'samples': ['Mustafa sahin', 'Emine Yildiz', 'Ali ozkan', 'Ibrahim Dogan', 'Zeynep Kilic', 'Huseyin Aslan', 'Zeliha Polat', 'Osman Akin']},
    {'length': 14, 'samples': ['Hatice Arslan', 'Suleyman Yucel', 'Hacer Turk', 'Meryem ozturk', 'Yasar Tunc', 'Rukiye Aydin']},
    {'length': 15, 'samples': ['Mehmet Demir', 'Ibrahim ozkan', 'Abdullah Gunes', 'Abdullah Dogan']},
    {'length': 16, 'samples': ['Mustafa Yilmaz', 'Suleyman Erdogan', 'Recep ozdemir', 'Hanife Basaran', 'Yusuf Koc', 'Gulay Karaca', 'Hasan Ucar', 'Nuriye ciftci', 'Kemal Duman', 'Necla Keskin']},
    {'length': 17, 'samples': ['Ramazan Bulut', 'Melahat Tas', 'Bayram Yavuz', 'Saliha Ceylan', 'omer Sezer', 'Cemile Ergin', 'Halil Kurt', 'Fatih Uzun']}
]

adres_len_samples = [
    {'length': 3, 'samples': ['Ist', 'Ank']},
    {'length': 5, 'samples': ['Bursa', 'Izmir', 'Adana']},
    {'length': 6, 'samples': ['Ankara']},
    {'length': 7, 'samples': ['Antalya', 'Denizli']},
    {'length': 8, 'samples': ['Istanbul', 'Eskisehir']},
    {'length': 9, 'samples': ['Samsun', 'Gaziantep', 'Atakum']},
    {'length': 10, 'samples': ['Kocasinan', 'Bursa/Mrkz', 'Konak/Izm']},
    {'length': 11, 'samples': ['cankaya/Ank', 'Kadikoy/Ist']},
    {'length': 12, 'samples': ['No:15 Kadikoy', 'Cad. cankaya']},
    {'length': 13, 'samples': ['Mah. Inonu Cad']},
    {'length': 14, 'samples': ['Zafer Mah. Kordon', 'Ataturk Mah. Cad']},
    {'length': 15, 'samples': ['Sok. No:23 Bursa']},
    {'length': 16, 'samples': ['Fatih Mah. Sinan S.']},
    {'length': 17, 'samples': ['Kultur Mah. Hurriyet']},
    {'length': 18, 'samples': ['Gazi Bulvari No:42']},
    {'length': 19, 'samples': ['Yildirim Mah. Istiklal']},
    {'length': 20, 'samples': ['Barbaros Mah. Ataturk']},
    {'length': 21, 'samples': ['Guzelyali Mah. sehit C.', 'Kocatepe Mah. Mithatp.']},
    {'length': 22, 'samples': ['Fevzi Pasa Mah. Gaziler', 'Mehmet Akif Mah. can.']},
    {'length': 23, 'samples': ['Yenisehir Mah. Cumhur. B.']},
    {'length': 24, 'samples': ['Selcuklu Mah. Mevlana Bul.', 'Inonu Mah. Ataturk Cad. ']},
    {'length': 25, 'samples': ['camlik Mah. Sahil Yolu No:', 'Yesilova Mah. Ankara Cad. ']},
    {'length': 26, 'samples': ['Karsiyaka Mah. Izmir Yolu N', 'Bati Mah. Londra Asfalti No']},
    {'length': 27, 'samples': ['Dogu Mah. Bagdat Cad. No:2', 'Merkez Mah. Sakarya Cad. No']},
    {'length': 28, 'samples': ['Guney Mah. Antalya Bulvari ', 'Kuzey Mah. Sivas Yolu No:87', 'Anadolu Mah. Trabzon Cad. N', 'Rumeli Mah. Edirne Yolu No:']},
    {'length': 29, 'samples': ['Orta Mah. Konya Cad. No:54 ', 'Ic Mah. Ankara Bulvari No:76', 'Dis Mah. Istanbul Yolu No:1']},
    {'length': 30, 'samples': ['Alt Mah. Bursa Cad. No:43 Os', 'ust Mah. Adana Yolu No:85 S', 'on Mah. Mersin Bulvari No:16']},
    {'length': 31, 'samples': ['Arka Mah. Gaziantep Cad. No', 'Orman Mah. Tabiat Yolu No:18', 'Deniz Mah. Sahil Cad. No:7 K', 'Dag Mah. Uludag Sok. No:31 O', 'Vadi Mah. camlik Cad. No:64 ']},
    {'length': 32, 'samples': ['Tepe Mah. Yuksek Sok. No:19 c', 'Duz Mah. Ovacik Yolu No:145 ', 'Egik Mah. Yamac Cad. No:52 Ka']},
    {'length': 33, 'samples': ['Genis Mah. Acik Alan Sok. No:', 'Dar Mah. Kisa Yol No:6 Kadiko']},
    {'length': 34, 'samples': ['Uzun Mah. Mesafe Cad. No:174 ', 'Kisa Mah. Yakin Sok. No:11 Ko']},
    {'length': 35, 'samples': ['Buyuk Mah. Genis Bulvari No:2', 'Kucuk Mah. Minik Sok. No:4 Be']},
    {'length': 39, 'samples': ['Eski Mah. Tarih Cad. No:99 Ulus/Ankara']},
    {'length': 41, 'samples': ['Zafer Mah. Kordon Cad. No:8 Konak/Izmir']},
    {'length': 44, 'samples': ['Yeni Mah. Modern Bulvari No:156 Bornova/Izmir']},
    {'length': 46, 'samples': ['Ataturk Mah. Inonu Cad. No:15 Kadikoy/Istanbul']},
    {'length': 48, 'samples': ['Fatih Mah. Mimar Sinan Sok. No:23 Osmangazi/Bursa']},
    {'length': 50, 'samples': ['Cumhuriyet Mah. Gazi Bulvari No:42 cankaya/Ankara']},
    {'length': 51, 'samples': ['Kultur Mah. Hurriyet Cad. No:12 Seyhan/Adana']},
    {'length': 52, 'samples': ['Yildirim Mah. Istiklal Sok. No:5 Merkez/Eskisehir']},
    {'length': 53, 'samples': ['Guzelyali Mah. sehit Cad. No:34 Atakum/Samsun']},
    {'length': 56, 'samples': ['Barbaros Mah. Ataturk Bulvari No:67 Muratpasa/Antalya']},
    {'length': 57, 'samples': ['Kocatepe Mah. Mithatpasa Cad. No:56 Kocasinan/Kayseri']},
    {'length': 58, 'samples': ['Mehmet Akif Mah. cankiri Cad. No:89 Altindag/Ankara']},
    {'length': 59, 'samples': ['Fevzi Pasa Mah. Gaziler Cad. No:91 sahinbey/Gaziantep']},
    {'length': 60, 'samples': ['Yenisehir Mah. Cumhuriyet Bulvari No:78 Pamukkale/Denizli']},
    {'length': 61, 'samples': ['Selcuklu Mah. Mevlana Bulvari No:25 Selcuklu/Konya', 'Inonu Mah. Ataturk Cad. No:47 Tepebasi/Eskisehir']},
    {'length': 62, 'samples': ['camlik Mah. Sahil Yolu No:13 Bodrum/Mugla', 'Yesilova Mah. Ankara Cad. No:62 Osmangazi/Bursa']},
    {'length': 63, 'samples': ['Karsiyaka Mah. Izmir Yolu No:38 Torbali/Izmir', 'Bati Mah. Londra Asfalti No:155 Esenyurt/Istanbul', 'Dogu Mah. Bagdat Cad. No:244 Maltepe/Istanbul', 'Merkez Mah. Sakarya Cad. No:71 Adapazari/Sakarya']},
    {'length': 64, 'samples': ['Guney Mah. Antalya Bulvari No:103 Kepez/Antalya', 'Kuzey Mah. Sivas Yolu No:87 Melikgazi/Kayseri', 'Anadolu Mah. Trabzon Cad. No:29 Atakum/Samsun', 'Rumeli Mah. Edirne Yolu No:126 Arnavutkoy/Istanbul', 'Orta Mah. Konya Cad. No:54 Selcuklu/Konya', 'Ic Mah. Ankara Bulvari No:76 cankaya/Ankara', 'Dis Mah. Istanbul Yolu No:198 Nilufer/Bursa', 'Alt Mah. Bursa Cad. No:43 Osmangazi/Bursa', 'ust Mah. Adana Yolu No:85 Seyhan/Adana', 'on Mah. Mersin Bulvari No:167 Mezitli/Mersin', 'Arka Mah. Gaziantep Cad. No:92 sehitkamil/Gaziantep', 'Orman Mah. Tabiat Yolu No:18 Beykoz/Istanbul', 'Deniz Mah. Sahil Cad. No:7 Kartal/Istanbul', 'Dag Mah. Uludag Sok. No:31 Osmangazi/Bursa', 'Vadi Mah. camlik Cad. No:64 Bornova/Izmir', 'Tepe Mah. Yuksek Sok. No:19 cankaya/Ankara', 'Duz Mah. Ovacik Yolu No:145 Tuzla/Istanbul', 'Egik Mah. Yamac Cad. No:52 Karsiyaka/Izmir', 'Genis Mah. Acik Alan Sok. No:28 Nilufer/Bursa', 'Dar Mah. Kisa Yol No:6 Kadikoy/Istanbul', 'Uzun Mah. Mesafe Cad. No:174 cankaya/Ankara', 'Kisa Mah. Yakin Sok. No:11 Konak/Izmir', 'Buyuk Mah. Genis Bulvari No:233 Osmangazi/Bursa', 'Kucuk Mah. Minik Sok. No:4 Beyoglu/Istanbul', 'Guzel Mah. Hos Sok. No:37 Nilufer/Bursa', 'cirkin Mah. Kotu Yol No:83 Pendik/Istanbul', 'Temiz Mah. Pak Cad. No:21 Kecioren/Ankara', 'Kirli Mah. Pis Sok. No:77 Kartal/Istanbul', 'Sessiz Mah. Sakin Yolu No:14 cankaya/Ankara', 'Gurultulu Mah. samata Cad. No:118 Kadikoy/Istanbul', 'Aydinlik Mah. Isik Bulvari No:65 Konak/Izmir', 'Karanlik Mah. Golge Sok. No:39 Besiktas/Istanbul']}
]