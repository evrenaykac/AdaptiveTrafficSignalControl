# PPO Dosya Düzenleme Görevleri

Bağımsız ve temiz bir PPO çalışma ortamı yaratmak için aşağıdaki alt görevler yapılmıştır.

## 1. Klasör Yapısının Oluşturulması
- [x] `ppo_workspace/core` klasörünün oluşturulması
- [x] `ppo_workspace/scripts` klasörünün oluşturulması
- [x] `ppo_workspace/bat` klasörünün oluşturulması
- [x] `ppo_workspace/docs` klasörünün oluşturulması (MD dosyaları için)

## 2. Dosyaların Taşınması
- [x] `ppo_agent.py` ve `ppo_multi_agent.py` dosyalarının `core/` klasörüne taşınması.
- [x] `train_ppo_3x3.py` ve `eval_ppo_3x3.py` dosyalarının `scripts/` klasörüne taşınması.
- [x] `40_ppo_3x3_train.bat` ve `41_ppo_3x3_eval.bat` dosyalarının `bat/` klasörüne taşınması.
- [x] PPO ile ilgili `.md` dosyalarının `docs/` klasörüne kopyalanması.

## 3. İçeriklerin/Import Yollarının Güncellenmesi
- [x] `scripts/train_ppo_3x3.py` dosyasındaki python import yollarının güncellenmesi (`from ppo_workspace.core...`).
- [x] `scripts/eval_ppo_3x3.py` dosyasındaki python import yollarının güncellenmesi.
- [x] `bat/40_ppo_3x3_train.bat` dosyasındaki script yolunun güncellenmesi (`py ppo_workspace\scripts\train_ppo_3x3.py` vb.).
- [x] `bat/41_ppo_3x3_eval.bat` dosyasındaki script yolunun güncellenmesi.
