(function(){
  const API_BASE = 'http://localhost:8001';
  const form = document.getElementById('predict-form');
  const textarea = document.getElementById('news-text');
  const resultCard = document.getElementById('result-card');
  const predLabel = document.getElementById('pred-label');
  const predConf = document.getElementById('pred-conf');
  const confBar = document.getElementById('confidence-bar');
  const loading = document.getElementById('loading');
  const errorBox = document.getElementById('error-box');
  const clearBtn = document.getElementById('clear-btn');
  const langBadge = document.getElementById('lang-badge');
  const langPill = document.getElementById('lang-pill');

  function detectLang(text){
    // simple heuristic: if any Devanagari characters, mark Hindi
    const devanagari = /[\u0900-\u097F]/;
    if(devanagari.test(text)) return 'Hindi';
    return 'English';
  }

  async function checkHealth(){
    try {
      const r = await fetch(API_BASE + '/health');
      const j = await r.json();
      return j;
    } catch(e){
      return { status: 'error', model_loaded: false };
    }
  }

  function setLoading(v){
    loading.style.display = v ? 'block' : 'none';
  }

  function setError(msg){
    if(!msg){
      errorBox.style.display = 'none';
      errorBox.textContent = '';
    } else {
      errorBox.style.display = 'block';
      errorBox.textContent = msg;
    }
  }

  function showResult(label, confidence, lang){
    predLabel.textContent = label;
    const pct = Math.round(confidence * 100);
    predConf.textContent = pct + '%';
    confBar.style.width = pct + '%';
    confBar.className = 'progress-bar ' + (label.toLowerCase().includes('fake') ? 'bg-danger' : 'bg-success');
    langPill.textContent = lang;
    resultCard.style.display = 'block';
  }

  form.addEventListener('submit', async function(e){
    e.preventDefault();
    setError('');
    resultCard.style.display = 'none';
    const text = (textarea.value || '').trim();
    if(!text){
      setError('Please paste some news text.');
      return;
    }
    const lang = detectLang(text);
    langBadge.textContent = 'Language: ' + lang;
    setLoading(true);
    try {
      const res = await fetch(API_BASE + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if(!res.ok){
        const t = await res.text();
        throw new Error('API error ' + res.status + ': ' + t);
      }
      const data = await res.json();
      showResult(data.label, data.confidence, lang);
    } catch(err){
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  });

  clearBtn.addEventListener('click', function(){
    textarea.value = '';
    resultCard.style.display = 'none';
    setError('');
  });

  // initial health check to guide the user
  (async function(){
    const h = await checkHealth();
    if(h.status !== 'ok'){
      setError('Backend not reachable at ' + API_BASE + '. Start the FastAPI server.');
    } else if(!h.model_loaded){
      setError('Model not loaded. Please run the training script first.');
    }
  })();
})();
