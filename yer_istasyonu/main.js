'use strict';

const { app, BrowserWindow, Menu, shell, ipcMain, clipboard } = require('electron');
const path = require('path');
const fs = require('fs');

// Aynı anda iki pencere açılıp ikisi de Jetson'a bağlanmasın
if (!app.requestSingleInstanceLock()) {
  app.quit();
}

let win = null;

function kayitKlasoru() {
  return path.join(app.getPath('pictures'), 'Sancak Yer Istasyonu');
}

function pencereOlustur() {
  win = new BrowserWindow({
    width: 1280,
    height: 900,
    minWidth: 820,
    minHeight: 600,
    backgroundColor: '#ffffff',
    title: 'Sancak İHA Yer İstasyonu',
    icon: path.join(__dirname, 'build', 'icon.png'),
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // Pencere arkada/küçültülmüşken de telemetri ve grafikler akmaya devam etsin
      backgroundThrottling: false,
    },
  });

  win.loadFile(path.join(__dirname, 'src', 'index.html'));
  win.once('ready-to-show', () => win.show());

  // "Haritada aç" gibi dış linkler sistem tarayıcısında açılsın
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (e, url) => {
    if (!url.startsWith('file://')) e.preventDefault();
  });

  win.on('closed', () => { win = null; });
}

function menuKur() {
  const sablon = [
    {
      label: 'Dosya',
      submenu: [
        {
          label: 'Kayıtlı kareler klasörünü aç',
          click: async () => {
            await fs.promises.mkdir(kayitKlasoru(), { recursive: true });
            shell.openPath(kayitKlasoru());
          },
        },
        { type: 'separator' },
        { role: 'quit', label: 'Çıkış' },
      ],
    },
    {
      label: 'Görünüm',
      submenu: [
        { role: 'reload', label: 'Yenile' },
        { role: 'togglefullscreen', label: 'Tam ekran' },
        { type: 'separator' },
        { role: 'zoomIn', label: 'Yakınlaştır' },
        { role: 'zoomOut', label: 'Uzaklaştır' },
        { role: 'resetZoom', label: 'Normal boyut' },
        { type: 'separator' },
        { role: 'toggleDevTools', label: 'Geliştirici araçları' },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(sablon));
}

function ipcKur() {
  ipcMain.handle('kare-kaydet', async (_e, veri, ad) => {
    const klasor = kayitKlasoru();
    await fs.promises.mkdir(klasor, { recursive: true });
    let dosya = String(ad || 'kare').replace(/[^\w.-]/g, '_');
    if (!dosya.toLowerCase().endsWith('.jpg')) dosya += '.jpg';
    const tamYol = path.join(klasor, dosya);
    await fs.promises.writeFile(tamYol, Buffer.from(veri));
    return tamYol;
  });

  ipcMain.handle('panoya-yaz', (_e, metin) => {
    clipboard.writeText(String(metin));
    return true;
  });
}

app.on('second-instance', () => {
  if (win) {
    if (win.isMinimized()) win.restore();
    win.focus();
  }
});

app.whenReady().then(() => {
  menuKur();
  ipcKur();
  pencereOlustur();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) pencereOlustur();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});