# GitHub, LinkedIn ve Koç Üniversitesi Başvuru Paketi

Bu metinler kopyalanıp kullanılmak üzere hazırlanmıştır. Ölçümler projenin
ayrılmış test setindeki gerçek sonuçlardır; iddialar abartılmamıştır.

## 1. GitHub deposu

**Önerilen depo adı**  
`industrial-predictive-maintenance-pytorch`

**GitHub “About” açıklaması**  
`Leakage-safe industrial failure prediction with PyTorch, calibrated probabilities, automated tests and a Streamlit demo.`

**Konular / Topics**  
`pytorch`, `deep-learning`, `predictive-maintenance`, `industrial-ai`,
`imbalanced-classification`, `streamlit`, `model-calibration`, `data-science`

**Kısa Türkçe açıklama**  
Endüstriyel sensör verilerinden makine arızası riskini tahmin eden; veri
sızıntısı kontrolü, dengesiz sınıf yönetimi, olasılık kalibrasyonu, otomatik
testler ve Streamlit demosu içeren uçtan uca PyTorch projesi.

**Depoyu oluştururken seçimler**

- Görünürlük: `Public`
- README ekleme: Hayır; pakette hazırdır.
- `.gitignore` ekleme: Hayır; pakette hazırdır.
- Lisans ekleme: Hayır; MIT lisansı pakette hazırdır.

**İlk yükleme komutları**

```bash
git init
git add .
git commit -m "Build end-to-end PyTorch predictive maintenance demo"
git branch -M main
git remote add origin https://github.com/halilsirvan/industrial-predictive-maintenance-pytorch.git
git push -u origin main
```

Şifre yerine GitHub’ın önerdiği kimlik doğrulama yöntemi kullanılmalıdır.

## 2. LinkedIn “Projeler” bölümü

**Proje adı**  
Industrial Predictive Maintenance with PyTorch

**İlişkili alanlar**  
Artificial Intelligence · Data Science · Industrial Engineering · Python

**İngilizce proje açıklaması**

> Developed an end-to-end industrial failure prediction system using PyTorch
> and 10,000 operating records from the UCI AI4I dataset. Built a leakage-safe
> preprocessing pipeline, engineered physically meaningful features, addressed
> a 3.39% rare-event class with weighted loss, calibrated model probabilities,
> and selected the decision threshold exclusively on validation data. The held-
> out test set achieved 0.974 ROC-AUC, 0.767 PR-AUC and 82.4% failure recall.
> Delivered automated tests, reproducible training artifacts and an interactive
> Streamlit demonstration.

**Türkçe proje açıklaması**

> UCI AI4I veri setindeki 10.000 endüstriyel çalışma kaydını kullanarak uçtan
> uca PyTorch arıza tahmin sistemi geliştirdim. Veri sızıntısını önleyen bir
> ön işleme hattı kurdum, fiziksel anlamlı özellikler türettim, yalnızca %3,39
> oranındaki arıza sınıfını ağırlıklı kayıpla ele aldım, model olasılıklarını
> kalibre ettim ve karar eşiğini yalnızca doğrulama verisiyle belirledim.
> Ayrılmış test setinde 0,974 ROC-AUC, 0,767 PR-AUC ve %82,4 arıza yakalama
> oranı elde ettim. Otomatik testleri, yeniden üretilebilir model dosyalarını
> ve çalışan Streamlit demosunu tamamladım.

## 3. LinkedIn paylaşımı

> Endüstriyel bilgi sistemleri ve veri analizi deneyimimi yapay zekâ ile
> birleştiren yeni portföy projemi tamamladım: **Industrial Predictive
> Maintenance with PyTorch**.
>
> Projede 10.000 çalışma kaydından oluşan UCI AI4I veri setini kullandım.
> Arızalar verinin yalnızca %3,39’unu oluşturduğu için klasik doğruluk oranı
> yerine PR-AUC, arıza precision/recall ve F1 ölçütlerine odaklandım.
>
> Öne çıkan çalışmalar:
>
> • Hedef ve arıza-türü sütunlarını dışarıda bırakan veri sızıntısı kontrolü  
> • PyTorch MLP ve ağırlıklı kayıp fonksiyonu  
> • Olasılık kalibrasyonu ve doğrulama setinde eşik seçimi  
> • Otomatik testler ve GitHub Actions  
> • Çalışan Streamlit arayüzü
>
> Ayrılmış test seti sonucu: **0,974 ROC-AUC, 0,767 PR-AUC ve %82,4 arıza
> yakalama oranı**. Model 51 gerçek arızanın 42’sini yakaladı.
>
> Bu çalışma, üretim ortamında gerçek sensör verileri, zaman bazlı doğrulama ve
> veri kayması takibiyle genişletilebilecek bir portföy demosudur.
>
> GitHub: https://github.com/halilsirvan/industrial-predictive-maintenance-pytorch
>
> #PyTorch #ArtificialIntelligence #PredictiveMaintenance #IndustrialAI
> #DataScience #MachineLearning #Streamlit

## 4. CV’ye eklenecek proje

**İngilizce**

> **Industrial Predictive Maintenance with PyTorch — Independent Project,
> 2026:** Built an end-to-end failure-risk system on 10,000 industrial sensor
> records; prevented target leakage, handled a 3.4% rare-event class with
> weighted loss, calibrated probabilities, and achieved 0.974 ROC-AUC / 0.767
> PR-AUC with 82.4% failure recall on a held-out test set. Delivered automated
> tests, GitHub Actions and a Streamlit demo.

**Türkçe**

> **PyTorch ile Endüstriyel Kestirimci Bakım — Bağımsız Proje, 2026:** 10.000
> sensör kaydı üzerinde uçtan uca arıza riski sistemi geliştirdi; hedef
> sızıntısını önledi, %3,4 oranındaki nadir arıza sınıfını ağırlıklı kayıpla
> modelledi, olasılık kalibrasyonu uyguladı ve ayrılmış test setinde 0,974
> ROC-AUC, 0,767 PR-AUC ve %82,4 arıza yakalama oranına ulaştı. Otomatik test,
> GitHub Actions ve Streamlit demosu hazırladı.

## 5. Koç Üniversitesi başvuru formuna eklenecek İngilizce paragraf

> My background combines industrial engineering, an M.Sc. in Computer
> Engineering, information systems experience in a large process-industry
> organization, and long-term responsibility for data analysis and reporting.
> To demonstrate my current hands-on AI capability, I recently developed an
> end-to-end predictive-maintenance project in PyTorch. I designed a leakage-
> safe pipeline for an imbalanced industrial dataset, engineered domain-
> meaningful features, calibrated the neural network outputs, evaluated the
> model with rare-event metrics, and delivered a tested Streamlit application.
> This work reflects the contribution I hope to make at KUIS AI: connecting
> industrial needs with careful data analysis, reproducible software and
> practical machine-learning solutions.

## 6. Kısa işe alım uzmanı mesajı

> Hello, I have applied for the Research Engineer position at KUIS AI. My
> background includes an M.Sc. in Computer Engineering, industrial information
> systems experience at PETKİM, and extensive data-analysis and project work.
> I have also completed a PyTorch predictive-maintenance project with a
> reproducible pipeline, calibrated outputs, automated tests and a Streamlit
> demo. I would be pleased to discuss how this combination of industrial and
> technical experience could support KUIS AI’s academic–industry projects.

## 7. Kullanım sırası

1. Paketi GitHub’da `Public` depo olarak yayınlayın.
2. README’deki grafiklerin göründüğünü kontrol edin.
3. GitHub Actions sekmesinde testin yeşil sonuçlandığını doğrulayın.
4. Depo bağlantısını LinkedIn Projeler bölümüne ekleyin.
5. İngilizce CV’ye yukarıdaki tek proje maddesini koyun.
6. Koç Üniversitesi başvurusunda GitHub bağlantısını ve İngilizce paragrafı
   kullanın.
7. LinkedIn paylaşımını başvurudan sonra yayınlayın; böylece profilinizi
   inceleyen kişi doğrudan çalışan kanıta ulaşır.

