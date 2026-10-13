# Mülakat Anlatım Notu — Endüstriyel Arıza Tahmini

## 30 saniyelik anlatım

“PETKİM’deki endüstriyel bilgi sistemleri geçmişimle veri bilimi ve yapay zekâyı
birleştirmek için PyTorch tabanlı bir kestirimci bakım projesi geliştirdim. Veri
setinde arızalar yalnızca yüzde 3,39 olduğu için doğruluk oranına güvenmedim;
PR-AUC, hassasiyet ve arıza yakalama oranını izledim. Arıza türü sütunlarının
hedef bilgisini sızdıracağını fark ederek bunları modelden çıkardım. Model,
görmediği test verisinde 0,974 ROC-AUC ve 0,767 PR-AUC elde etti; 51 arızanın
42’sini yakaladı. Sonucu çalışan bir Streamlit arayüzüne dönüştürdüm.”

## Neden PyTorch kullandım?

- Model mimarisini, kayıp fonksiyonunu ve eğitim döngüsünü açıkça kontrol etmek.
- Dengesiz sınıf için ağırlıklı ikili çapraz entropi kullanmak.
- Modelin girdilere duyarlılığını gradyanlarla incelemek.
- Eğitim ve çıkarım bileşenlerini gerçek bir yazılım projesi gibi ayırmak.

## Sorulabilecek teknik sorular

**Neden doğruluk oranını ana ölçüt seçmediniz?**  
Verinin %96,61’i normal. Her kayda “normal” diyen işe yaramaz bir model bile
%96,61 doğruluk verir. Bu nedenle PR-AUC ve arıza sınıfı recall/F1 kullandım.

**Veri sızıntısını nasıl önlediniz?**  
TWF, HDF, PWF, OSF ve RNF sütunları hedef arızanın alt nedenlerini gösteriyor.
Bunları ve kimlik sütunlarını model girdilerinden açıkça çıkardım. Bu davranışı
birim testiyle de korudum.

**Eşik neden 0,50 değil?**  
Nadir olaylarda 0,50 keyfi kalabilir. Karar eşiğini test setine bakmadan,
doğrulama setinde arıza sınıfı F1 değerini en yüksek yapan noktadan seçtim.

**Olasılıkları neden kalibre ettiniz?**  
Ağırlıklı kayıp fonksiyonu sıralama performansını iyileştirebilir ancak ham
sinir ağı çıktıları gerçek olasılık gibi davranmayabilir. Doğrulama çıktıları
üzerinde Platt kalibrasyonu kullandım ve Brier skorunu raporladım.

**Model üretime hazır mı?**  
Hayır; veri sentetik ve tekil gözlemdeki arızayı sınıflandırıyor. Üretim için
zaman sıralı gerçek tesis verisi, makine bazlı ayrım, maliyet duyarlı eşik,
veri kayması takibi ve saha doğrulaması gerekir. Bunu açıkça README’de yazdım.

## Elde edilen sonuçlar

| Ölçüt | Sonuç |
|---|---:|
| ROC-AUC | 0,974 |
| PR-AUC | 0,767 |
| Arıza precision | 0,618 |
| Arıza recall | 0,824 |
| Arıza F1 | 0,706 |
| Yakalanan arıza | 42 / 51 |
| Yanlış alarm | 26 / 1.500 test kaydı |

## GitHub’a koymadan önce

1. `CITATION.cff` içindeki `USERNAME` alanını GitHub kullanıcı adınızla değiştirin.
2. Depo açıklaması: `Leakage-safe industrial failure prediction with PyTorch,
   calibrated probabilities, tests and a Streamlit demo.`
3. Konular: `pytorch`, `predictive-maintenance`, `deep-learning`,
   `industrial-ai`, `streamlit`, `imbalanced-classification`.
4. GitHub profilinizde depoyu sabitleyin.

