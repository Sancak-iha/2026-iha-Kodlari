'use strict';

// ======================================================================
// Ayarlar
// ======================================================================
const MAX_POINTS   = 120;    // grafiklerde tutulan nokta (CHART_MS ile ~60 sn)
const CHART_MS     = 500;    // grafiğe en fazla bu sıklıkla nokta ekle
const STALE_MS     = 3000;   // bu kadar veri gelmezse "veri gelmiyor"
const RECONNECT_MS = 2000;
const CAM_RETRY_MS = 3000;
const LOG_MAX      = 400;

const $ = (id) => document.getElementById(id);
const gcs = window.gcs || null;   // preload.js köprüsü (tarayıcıda açılırsa yok)

// ======================================================================
// DOM
// ======================================================================
const jetsonHost = $('jetsonHost');
const jetsonPort = $('jetsonPort');
const btnConnect = $('btnConnect');
const btnDisconnect = $('btnDisconnect');
const btnRefresh = $('btnRefresh');
const btnKareKaydet = $('btnKareKaydet');
const statusPill = $('statusPill');
const statusText = $('statusText');
const cameraStream = $('cameraStream');
const cameraPlaceholder = $('cameraPlaceholder');
const cameraUyari = $('cameraUyari');
const kameraBilgi = $('kameraBilgi');
const logKutu = $('logKutu');

// ======================================================================
// Kayıtlı ayarlar (son girilen IP/port hatırlanır)
// ======================================================================
function ayarOku() {
  try { return JSON.parse(localStorage.getItem('sancak-gcs') || '{}'); }
  catch (_) { return {}; }
}
function ayarYaz(a) {
  try { localStorage.setItem('sancak-gcs', JSON.stringify(a)); } catch (_) { /* yoksay */ }
}
{
  const a = ayarOku();
  const urlHost = new URLSearchParams(location.search).get('jetson');
  if (urlHost) jetsonHost.value = urlHost;
  else if (a.host) jetsonHost.value = a.host;
  if (a.port) jetsonPort.value = a.port;
}

// ======================================================================
// Grafikler
// ======================================================================
function makeChart(canvasId, label, color, opts) {
  if (typeof Chart === 'undefined') return null;     // chart.js yüklenemediyse uygulama yine çalışsın
  const ctx = $(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label, data: [],
        borderColor: color, backgroundColor: color + '22',
        borderWidth: 1.5, tension: 0.25, pointRadius: 0, fill: true, spanGaps: false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {
        legend: {
          display: true, position: 'top', align: 'start',
          labels: { color: '#ffffff', font: { size: 10 }, boxWidth: 12, boxHeight: 2 },
        },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.08)' },
          ticks: { color: '#888', font: { size: 9 }, maxTicksLimit: 6, maxRotation: 0 },
          border: { color: '#333' },
        },
        y: Object.assign({
          grid: { color: 'rgba(255,255,255,0.08)' },
          ticks: { color: '#888', font: { size: 9 } },
          border: { color: '#333' },
        }, opts),
      },
    },
  });
}

// Sabit max yerine suggestedMax: 100 m üstüne çıkınca grafik kesilmesin
const chartAlt     = makeChart('chartAlt', 'İrtifa (m)', '#5db3e9', { suggestedMin: 0, suggestedMax: 100 });
const chartSpeed   = makeChart('chartSpeed', 'Hız (m/s)', '#e878a8', { suggestedMin: 0, suggestedMax: 30 });
const chartBattery = makeChart('chartBattery', 'Batarya (%)', '#6dd490', { min: 0, max: 100 });

function pushChart(chart, value, ts) {
  if (!chart) return;
  chart.data.labels.push(ts);
  chart.data.datasets[0].data.push(sayiMi(value) ? value : null);
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}
function chartsTemizle() {
  for (const c of [chartAlt, chartSpeed, chartBattery]) {
    if (!c) continue;
    c.data.labels.length = 0;
    c.data.datasets[0].data.length = 0;
    c.update('none');
  }
}

// ======================================================================
// Yardımcılar
// ======================================================================
function sayiMi(v) { return typeof v === 'number' && Number.isFinite(v); }
function fmt(v, d = 2) { return sayiMi(v) ? v.toFixed(d) : '—'; }
function saatStr(d = new Date()) {
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((x) => String(x).padStart(2, '0')).join(':');
}
function degerYaz(el, v, unit, d = 2, sinif = '') {
  el.textContent = fmt(v, d);
  if (sayiMi(v) && unit) {
    const s = document.createElement('span');
    s.className = 'unit';
    s.textContent = unit;
    el.appendChild(s);
  }
  el.className = 'data-val' + (sinif ? ' ' + sinif : '');
}
function metinYaz(el, metin, sinif = '') {
  el.textContent = (metin === null || metin === undefined || metin === '') ? '—' : String(metin);
  el.className = 'data-val' + (sinif ? ' ' + sinif : '');
}
function modTemizle(m) {
  // MAVSDK FlightMode enum'u "FlightMode.MISSION" veya "MISSION" gelebilir
  return m ? String(m).replace(/^FlightMode\./, '') : null;
}

// ======================================================================
// Bağlantı durumu
// ======================================================================
let ws = null;
let wantConnected = false;
let reconnectTimer = null;
let lastMsgAt = 0;
let lastChartAt = 0;
let mjpegPort = 8080;
let streamPath = '/stream.mjpg';
let snapshotPath = '/snapshot.jpg';
let aktifHost = '';
let sonHedefAnahtar = null;

function pill(durum, metin) {
  statusPill.classList.remove('bagli', 'baglaniyor', 'uyari');
  if (durum) statusPill.classList.add(durum);
  statusText.textContent = metin;
}

function arayuzDurumu(durum) {
  // durum: 'bosta' | 'baglaniyor' | 'bagli' | 'yeniden'
  const mesgul = durum !== 'bosta';
  btnConnect.disabled = mesgul;
  btnDisconnect.disabled = !mesgul;        // yeniden denerken de iptal edilebilsin
  jetsonHost.disabled = mesgul;
  jetsonPort.disabled = mesgul;
  btnKareKaydet.disabled = durum !== 'bagli';
  if (durum === 'bosta') pill('', 'Bağlı Değil');
  if (durum === 'baglaniyor') pill('baglaniyor', 'Bağlanıyor...');
  if (durum === 'yeniden') pill('baglaniyor', 'Yeniden bağlanıyor...');
  if (durum === 'bagli') pill('bagli', 'Bağlı');
}

function degerleriSifirla() {
  for (const id of ['valPitch', 'valYaw', 'valRoll', 'valSpeed', 'valClimb', 'armStatus', 'valMode',
                    'valAlt', 'valBattery', 'valGps', 'valState', 'valWp', 'valSamples', 'valReject', 'valPos']) {
    metinYaz($(id), null);
  }
  kameraBilgi.textContent = '';
}

function connectWs() {
  const host = jetsonHost.value.trim();
  const port = jetsonPort.value.trim() || '8765';
  if (!host) { yerelLog('Jetson IP adresini girin.', 'ERROR'); jetsonHost.focus(); return; }
  if (!/^\d{1,5}$/.test(port)) { yerelLog('Port sadece rakam olmalı.', 'ERROR'); jetsonPort.focus(); return; }

  ayarYaz(Object.assign(ayarOku(), { host, port }));
  aktifHost = host;
  wantConnected = true;
  clearTimeout(reconnectTimer);
  if (ws) return;

  const url = `ws://${host}:${port}`;
  arayuzDurumu(reconnectTimer ? 'yeniden' : 'baglaniyor');
  reconnectTimer = null;

  let sok;
  try { sok = new WebSocket(url); }
  catch (e) { yerelLog('Adres hatalı: ' + url, 'ERROR'); arayuzDurumu('bosta'); wantConnected = false; return; }
  ws = sok;

  sok.onopen = () => {
    if (ws !== sok) return;
    lastMsgAt = Date.now();
    arayuzDurumu('bagli');
    yerelLog(`Bağlandı: ${url}`);
    startCamera();
  };

  sok.onmessage = (event) => {
    if (ws !== sok) return;
    let msg;
    try { msg = JSON.parse(event.data); } catch (_) { return; }
    lastMsgAt = Date.now();
    if (msg.type === 'hello') return handleHello(msg);
    if (msg.type === 'log') return handleLog(msg);
    return handleTelemetry(msg);   // type 'telemetry' veya eski sunucudan tipsiz paket
  };

  sok.onerror = () => { /* onclose zaten gelecek */ };

  sok.onclose = () => {
    if (ws !== sok) return;
    ws = null;
    stopCamera();
    if (wantConnected) {
      arayuzDurumu('yeniden');
      reconnectTimer = setTimeout(connectWs, RECONNECT_MS);
    } else {
      arayuzDurumu('bosta');
    }
  };
}

function disconnectWs() {
  wantConnected = false;
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  if (ws) {
    const sok = ws;
    ws = null;
    try { sok.close(); } catch (_) { /* yoksay */ }
  }
  stopCamera();
  arayuzDurumu('bosta');
  degerleriSifirla();
  yerelLog('Bağlantı kesildi.');
}

// Veri akışı bekçisi: soket açık ama paket gelmiyorsa operatör görsün
setInterval(() => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const bayat = Date.now() - lastMsgAt > STALE_MS;
  document.querySelectorAll('.data-table').forEach((t) => t.classList.toggle('bayat', bayat));
  if (bayat) pill('uyari', 'Veri gelmiyor');
}, 1000);

// ======================================================================
// Kamera (MJPEG)
// ======================================================================
let camRetryTimer = null;

function kameraUrl(yol) {
  return `http://${aktifHost}:${mjpegPort}${yol}?t=${Date.now()}`;
}

function startCamera() {
  clearTimeout(camRetryTimer);
  if (!aktifHost) return;
  cameraStream.onerror = () => {
    cameraStream.style.display = 'none';
    cameraPlaceholder.style.display = 'block';
    cameraPlaceholder.textContent = 'Kamera akışı yok, yeniden deneniyor...';
    clearTimeout(camRetryTimer);
    if (ws) camRetryTimer = setTimeout(startCamera, CAM_RETRY_MS);
  };
  cameraStream.src = kameraUrl(streamPath);
  cameraStream.style.display = 'block';
  cameraPlaceholder.style.display = 'none';
}

function stopCamera() {
  clearTimeout(camRetryTimer);
  cameraStream.onerror = null;
  cameraStream.removeAttribute('src');   // akışı gerçekten kapatır
  cameraStream.style.display = 'none';
  cameraPlaceholder.style.display = 'block';
  cameraPlaceholder.textContent = 'Bağlantı bekleniyor...';
  cameraUyari.hidden = true;
}

function refreshCamera() {
  if (ws && aktifHost) startCamera();
}

async function kareKaydet(otomatik = false) {
  if (!aktifHost) return;
  const ad = `${otomatik ? 'hedef' : 'kare'}_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
  try {
    const r = await fetch(kameraUrl(snapshotPath));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const veri = new Uint8Array(await r.arrayBuffer());
    if (gcs && gcs.kareKaydet) {
      const yol = await gcs.kareKaydet(veri, ad);
      yerelLog('Kare kaydedildi: ' + yol);
    } else {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([veri], { type: 'image/jpeg' }));
      a.download = ad;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    }
  } catch (e) {
    yerelLog('Kare kaydedilemedi: ' + e.message, 'ERROR');
  }
}

// ======================================================================
// Gelen mesajlar
// ======================================================================
function handleHello(h) {
  const eskiPort = mjpegPort;
  if (sayiMi(h.mjpeg_port)) mjpegPort = h.mjpeg_port;
  if (h.stream_path) streamPath = h.stream_path;
  if (h.snapshot_path) snapshotPath = h.snapshot_path;
  yerelLog(`Jetson: ${h.hostname || '?'} | kamera portu ${mjpegPort}, ` +
           `${h.stream_fps || '?'} fps, ${h.stream_width || '?'} px`);
  if (mjpegPort !== eskiPort) startCamera();
}

function handleTelemetry(t) {
  const now = Date.now();

  if (t.attitude) {
    degerYaz($('valPitch'), t.attitude.pitch, '°');
    degerYaz($('valYaw'), t.attitude.yaw, '°');
    degerYaz($('valRoll'), t.attitude.roll, '°');
  }
  if (t.position) {
    degerYaz($('valAlt'), t.position.rel_alt, 'm');
    metinYaz($('valPos'), sayiMi(t.position.lat) && (t.position.lat !== 0 || t.position.lon !== 0)
      ? `${t.position.lat.toFixed(6)}, ${t.position.lon.toFixed(6)}` : null);
  }
  if (t.speed) {
    degerYaz($('valSpeed'), t.speed.ground, 'm/s');
    degerYaz($('valClimb'), t.speed.climb, 'm/s');
  } else {
    metinYaz($('valSpeed'), null);
    metinYaz($('valClimb'), null);
  }
  if (t.battery && sayiMi(t.battery.remaining_pct)) {
    const p = t.battery.remaining_pct;
    const v = sayiMi(t.battery.voltage) ? `  ${t.battery.voltage.toFixed(1)} V` : '';
    metinYaz($('valBattery'), `%${p.toFixed(0)}${v}`, p < 20 ? 'val-kotu' : p < 40 ? 'val-uyari' : 'val-iyi');
  } else {
    metinYaz($('valBattery'), null);
  }
  if (t.gps) {
    const fix = t.gps.fix ? String(t.gps.fix).replace(/^FixType\./, '') : null;
    const sat = sayiMi(t.gps.sats) ? `${t.gps.sats} uydu` : '';
    metinYaz($('valGps'), [fix, sat].filter(Boolean).join(', '),
      sayiMi(t.gps.sats) ? (t.gps.sats >= 8 ? 'val-iyi' : 'val-uyari') : '');
  }

  if (t.status) {
    const s = t.status;
    metinYaz($('armStatus'), s.armed ? 'Armed' : 'Disarmed', s.armed ? 'val-kotu' : '');
    metinYaz($('valMode'), modTemizle(s.flight_mode));
    const durumSinif = s.state === 'HATA' ? 'val-kotu'
      : (s.state && s.state !== 'ARANIYOR') ? 'val-iyi' : '';
    let durumMetin = s.state;
    if (s.state === 'DUZ_GIDILIYOR' && sayiMi(s.lead_done)) {
      durumMetin += `  (${s.lead_done.toFixed(0)}/${fmt(s.lead_distance, 0)} m)`;
    }
    metinYaz($('valState'), durumMetin, durumSinif);
    let wp = sayiMi(s.mission_current) && s.mission_current >= 0
      ? `${s.mission_current} / ${s.mission_total}` : null;
    if (wp && sayiMi(s.target_wp)) wp += `  (hedef WP${s.target_wp})`;
    metinYaz($('valWp'), wp);
  }

  if (t.detection) {
    const d = t.detection;
    let ornek = `${d.samples ?? 0} / ${d.min_samples ?? '?'}`;
    if (sayiMi(d.depression)) ornek += `  (dep ${d.depression.toFixed(0)}°, ${fmt(d.range, 0)} m)`;
    metinYaz($('valSamples'), ornek);
    metinYaz($('valReject'), d.reject, d.reject ? 'val-uyari' : '');
  }

  hedefGuncelle(t.target);

  if (t.connected === false) pill('uyari', 'Otopilot bağlı değil');
  else if (t.telemetry_fresh === false) pill('uyari', 'Telemetri bayat');
  else if (ws) pill('bagli', 'Bağlı');

  if (t.video) {
    const v = t.video;
    kameraBilgi.textContent = sayiMi(v.fps) && v.fps > 0 ? `${v.fps.toFixed(0)} fps` : '';
    const donmus = sayiMi(v.age) && v.age > 3;
    cameraUyari.hidden = !donmus || cameraStream.style.display === 'none';
    if (donmus) cameraUyari.textContent = `Görüntü ${v.age.toFixed(0)} sn'dir yenilenmiyor`;
  }

  if (now - lastChartAt >= CHART_MS) {
    lastChartAt = now;
    const ts = saatStr();
    if (t.position) pushChart(chartAlt, t.position.rel_alt, ts);
    pushChart(chartSpeed, t.speed ? t.speed.ground : null, ts);
    pushChart(chartBattery, t.battery ? t.battery.remaining_pct : null, ts);
  }
}

function hedefGuncelle(h) {
  const kart = $('hedefKart');
  if (!h || !sayiMi(h.lat) || !sayiMi(h.lon)) {
    $('hedefBos').hidden = false;
    $('hedefIcerik').hidden = true;
    sonHedefAnahtar = null;
    return;
  }
  const anahtar = `${h.lat.toFixed(7)},${h.lon.toFixed(7)}`;
  $('hedefBos').hidden = true;
  $('hedefIcerik').hidden = false;
  $('hedefKoord').textContent = anahtar.replace(',', ', ');
  const parca = [];
  if (h.sinif) parca.push(`Sınıf: ${h.sinif}`);
  if (sayiMi(h.spread_m)) parca.push(`Hata payı: ±${h.spread_m.toFixed(0)} m`);
  if (sayiMi(h.samples)) parca.push(`${h.samples} örnek`);
  $('hedefDetay').textContent = parca.join('   ');
  $('hedefHarita').href = `https://www.google.com/maps?q=${h.lat.toFixed(7)},${h.lon.toFixed(7)}`;

  if (anahtar !== sonHedefAnahtar) {
    const ilk = sonHedefAnahtar === null;
    sonHedefAnahtar = anahtar;
    kart.classList.remove('yeni');
    void kart.offsetWidth;            // animasyonu yeniden başlat
    kart.classList.add('yeni');
    if (ilk) {
      yerelLog(`HEDEF: ${anahtar}`, 'HEDEF');
      kareKaydet(true);               // kilitlenme anının karesini otomatik sakla
    }
  }
}

// ======================================================================
// Olay günlüğü
// ======================================================================
function logSatiri(zaman, seviye, metin) {
  const satir = document.createElement('div');
  let sinif = seviye;
  if (/HEDEF (KILITLENDI|USTUNDE|WAYPOINT)|KOORDINAT:/.test(metin)) sinif = 'HEDEF';
  satir.className = 'log-satir log-' + sinif;
  const z = document.createElement('span');
  z.className = 'zaman';
  z.textContent = zaman;
  satir.appendChild(z);
  satir.appendChild(document.createTextNode(metin));

  const enAltta = logKutu.scrollHeight - logKutu.scrollTop - logKutu.clientHeight < 30;
  logKutu.appendChild(satir);
  while (logKutu.childElementCount > LOG_MAX) logKutu.firstElementChild.remove();
  if (enAltta) logKutu.scrollTop = logKutu.scrollHeight;
}

function handleLog(m) {
  for (const l of m.lines || []) {
    const zaman = sayiMi(l.t) ? saatStr(new Date(l.t * 1000)) : saatStr();
    logSatiri(zaman, l.level || 'INFO', String(l.text ?? ''));
  }
}

function yerelLog(metin, seviye = 'YEREL') {
  logSatiri(saatStr(), seviye, metin);
}

// ======================================================================
// Butonlar
// ======================================================================
btnConnect.addEventListener('click', () => { chartsTemizle(); connectWs(); });
btnDisconnect.addEventListener('click', disconnectWs);
btnRefresh.addEventListener('click', refreshCamera);
btnKareKaydet.addEventListener('click', () => kareKaydet(false));
$('btnLogTemizle').addEventListener('click', () => { logKutu.textContent = ''; });
for (const el of [jetsonHost, jetsonPort]) {
  el.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !btnConnect.disabled) btnConnect.click(); });
}

$('btnKopyala').addEventListener('click', async () => {
  const metin = $('hedefKoord').textContent;
  try {
    if (gcs && gcs.panoyaYaz) await gcs.panoyaYaz(metin);
    else await navigator.clipboard.writeText(metin);
    yerelLog('Koordinat panoya kopyalandı: ' + metin);
  } catch (e) {
    yerelLog('Kopyalanamadı: ' + e.message, 'ERROR');
  }
});

arayuzDurumu('bosta');
if (typeof Chart === 'undefined') {
  yerelLog('Chart.js yüklenemedi; grafikler kapalı. "npm install" çalıştırıldı mı?', 'ERROR');
}