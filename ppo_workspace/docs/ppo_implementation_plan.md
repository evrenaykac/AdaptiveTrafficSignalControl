# PPO Dosya Organizasyonu Planı

Mevcut durumda PPO dosyaları `ppo_workspace` klasörüne karışık olarak kaydedilmişti. Kullanıcının talebi üzerine karmaşayı önlemek ve her şeyi derli toplu tutmak için aşağıdaki yeni yapıya geçilmiştir:

## Klasör ve Dosya Yapısı

PPO ile ilgili tüm kodlar `ppo_workspace` klasörü altında toplanmış ve kendi içinde kategorize edilmiştir:

```text
SumoMainFunctions/
├── ppo_workspace/
│   ├── core/                  ← Ajan ve Kontrolcü Sınıfları
│   │   ├── ppo_agent.py
│   │   └── ppo_multi_agent.py
│   │
│   ├── scripts/               ← Eğitim ve Değerlendirme Python Kodları
│   │   ├── train_ppo_3x3.py
│   │   └── eval_ppo_3x3.py
│   │
│   ├── bat/                   ← Başlatıcı Bat Dosyaları
│   │   ├── 40_ppo_3x3_train.bat
│   │   └── 41_ppo_3x3_eval.bat
│   │
│   └── docs/                  ← PPO Dokümantasyonu (Bu dosya gibi)
```

## Yapılan Değişiklikler

1. **Dosya Taşıma:** Tüm PPO dosyaları yukarıdaki yapıya göre `ppo_workspace` içerisindeki alt klasörlere (`core`, `scripts`, `bat`, `docs`) taşındı.
2. **Import Yollarının Güncellenmesi:** 
   - `train_ppo_3x3.py` ve `eval_ppo_3x3.py` içindeki import kısımları yeni klasör yapısını yansıtacak şekilde (`from ppo_workspace.core.ppo_agent import PPOAgent` vb.) güncellendi.
3. **Bat Dosyalarının Güncellenmesi:** 
   - `bat` klasörü içindeki başlatıcı dosyalar, artık python scriptlerini `ppo_workspace/scripts/...` konumundan çağıracak şekilde güncellendi.

Bu yapı sayesinde Ana projenin `src`, `scripts` ve `tools/bat` klasörleri PPO kodlarıyla karışmamakta; tamamen bağımsız, modüler ve temiz bir PPO çalışma alanı elde edilmektedir.
