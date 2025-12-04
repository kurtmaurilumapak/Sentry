import { contextBridge } from 'electron';

// In this simple prototype we don't need custom Electron APIs yet,
// but we keep the bridge ready for future extensions (e.g. file system, config).
contextBridge.exposeInMainWorld('electronAPI', {
  // placeholder for future APIs
});


