# Anlam Atölyesi

Türkçe sözcükler arasındaki anlamsal ilişkileri **Gensim Word2Vec** ile keşfetmeye yarayan, **FastAPI** tabanlı web uygulaması.

## Özellikler

- Bir sözcüğe en yakın sözcükleri bulma
- İki sözcük arasındaki kosinüs benzerliğini ölçme
- Sözcük analojileri (`kral + kadın − erkek`) kurma
- Cümleleri ortalama sözcük vektörleriyle karşılaştırma
- Güvenlik başlıklarıyla korunan statik API belgeleri (`/docs`)
- Vercel için sıfır yapılandırmaya yakın dağıtım

## Mimari

```text
Tarayıcı → FastAPI REST API → Gensim Word2Vec → Türkçe eğitim derlemi
```

Model `scripts/train_model.py` ile `data/turkish_corpus.txt` dosyasından Gensim Word2Vec kullanılarak eğitilir. Üretilen vektörler, sunucusuz ortamda hızlı açılış için uygulama tarafından salt okunur biçimde yüklenir. Bu küçük veri kümesi eğitim/demonstrasyon amaçlıdır; üretim doğruluğu iddiası taşımaz.

## Yerelde çalıştırma

Python 3.11 veya üzeri gerekir.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train_model.py
uvicorn app.main:app --reload
```

Ardından `http://127.0.0.1:8000` adresini açın. API belgeleri `http://127.0.0.1:8000/docs` adresindedir.

## Test

```bash
pytest
```

## Vercel'e yayınlama

Vercel, `pyproject.toml` içindeki `app.main:app` giriş noktasını kullanır.

1. Bu projeyi GitHub'da yeni bir depoya gönderin.
2. Vercel'de **Add New → Project** ile depoyu içe aktarın.
3. Framework ayarını **Other** olarak bırakıp **Deploy** seçin.

CLI ile alternatif:

```bash
npx vercel
npx vercel --prod
```

## API örneği

```bash
curl "http://127.0.0.1:8000/api/nearest?word=bilim&limit=5"
```

## Teknolojiler

- Python, FastAPI, Uvicorn
- Gensim Word2Vec
- NumPy / SciPy
- HTML, CSS, JavaScript

## Lisans

MIT
