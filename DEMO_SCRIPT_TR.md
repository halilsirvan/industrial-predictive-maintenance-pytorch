# İki Dakikalık Canlı Demo ve Mülakat Metni

## 0:00–0:20 — Problem

“Bu projede endüstriyel çalışma koşullarından makine arızası riskini tahmin
ediyorum. Veri setinde arıza oranı yalnızca %3,39. Bu nedenle yüksek doğruluk
tek başına anlamlı değil; gerçek arızaları yakalamak ve yanlış alarmı sınırlamak
gerekiyor.”

## 0:20–0:45 — Veri ve sızıntı kontrolü

README’de veri yapısını gösterin.

“Veri setindeki beş arıza-türü sütunu hedefi oluşturan bilgileri içeriyor.
Bunları modele vermek veri sızıntısı yaratırdı. Bu sütunları, kimlikleri ve
hedefi özelliklerden çıkardım; ayrıca bunu otomatik testle korudum.”

## 0:45–1:10 — Model

“Sayısal alanları standartlaştırdım, ürün türünü one-hot kodladım. Sıcaklık
farkı, mekanik güç ve tork-aşınma etkileşimi olmak üzere üç fiziksel anlamlı
özellik oluşturdum. PyTorch’ta 32 ve 16 nöronlu küçük bir ağ, ağırlıklı BCE
kaybı, AdamW ve dropout kullandım. Ham skorları doğrulama verisi üzerinde
Platt yöntemiyle kalibre ettim.”

## 1:10–1:35 — Sonuç

Değerlendirme grafiğini gösterin.

“Modelin hiç görmediği 1.500 test kaydında ROC-AUC 0,974 ve PR-AUC 0,767 oldu.
51 arızanın 42’sini yakaladı; arıza recall değeri %82,4. Eşiği test verisine
göre değil, doğrulama setinde F1’i en iyi yapan noktadan seçtim.”

## 1:35–1:55 — Uygulama

Streamlit’te önce varsayılan değerlerle tahmin yapın; sonra şu yüksek risk
örneğini girin:

- Type: `L`
- Air temperature: `303 K`
- Process temperature: `310 K`
- Rotational speed: `1250 rpm`
- Torque: `65 Nm`
- Tool wear: `220 min`

“Arayüz, kalibre edilmiş risk tahmini ve bakım alarmı gösteriyor. Model ve
ön işleme dosyaları eğitimden bağımsız biçimde yükleniyor.”

## 1:55–2:10 — Sınırlar

“Bunu üretime hazırmış gibi sunmuyorum. Veri sentetik ve mevcut gözlemdeki
arızayı sınıflandırıyor. Sonraki adım gerçek tesis verisiyle zaman bazlı
doğrulama, maliyet duyarlı eşik ve veri kayması takibi olurdu.”

## Kritik mülakat cümlesi

“Benim katkım yalnız model eğitmek değil; endüstriyel problemi doğru ölçüte
çevirmek, veri sızıntısını engellemek, sonucu yeniden üretilebilir yazılıma ve
iş kararını destekleyen bir arayüze dönüştürmektir.”

