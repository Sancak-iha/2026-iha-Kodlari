'use strict';

// Arayüze sadece bu iki işlev açılır; Node.js'e doğrudan erişim yok.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gcs', {
  masaustu: true,
  kareKaydet: (veri, ad) => ipcRenderer.invoke('kare-kaydet', veri, ad),
  panoyaYaz: (metin) => ipcRenderer.invoke('panoya-yaz', metin),
});