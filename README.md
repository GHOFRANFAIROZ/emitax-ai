# Emitax — AI Verification Layer

طبقة تدقيق البيانات (ذكاء اصطناعي) لمشروع Emitax — TEKNOFEST 2026 / BlockSec26.

## التشغيل (محلياً — VSCode)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# ضعي ملف CEMS داخل data/raw/  ثم:
python run_prepare.py     # تحميل + تجميع شهري + حفظ المخرجات
python run_eda.py         # رسوم EDA في docs/eda/
pytest -q                 # الاختبارات
```

## الملفّات
- `configs/default.yaml` — كل الإعدادات (أسماء الأعمدة، الغازات، فترة التجميع). عدّلي هنا فقط لو تغيّرت الداتا.
- `src/emitax/data/loader.py` — تحميل CEMS محايد للأعمدة/الحجم (يدعم القراءة على دفعات).
- `src/emitax/data/aggregate.py` — تجميع شهري (= القياس المرجعي للفحص ١) + كشف توفّر القياس.
- `run_prepare.py` / `run_eda.py` — سكربتات المرحلة ١.

## المرونة (Scalability)
كل شيء مدفوع بالإعدادات؛ اللودر يتجاهل الأعمدة الزائدة ويعمل مع أي عدد وحدات/مصانع.
للملفّات الضخمة: اضبطي `data.chunksize` في الإعدادات.
