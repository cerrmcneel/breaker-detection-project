/**
 * captureHandoff.js
 * Shared IndexedDB handoff module connecting PanelSafe Live Viewfinder and Analysis flow.
 * Database: 'panelsafe-handoff', version 1, store: 'captures', key: 'pending'
 */

const DB_NAME = 'panelsafe-handoff';
const DB_VERSION = 1;
const STORE_NAME = 'captures';
const PENDING_KEY = 'pending';
const MAX_AGE_MS = 15 * 60 * 1000; // 15 minutes TTL

function openDB() {
    return new Promise((resolve, reject) => {
        if (!('indexedDB' in window)) {
            return reject(new Error('IndexedDB not supported in this browser'));
        }
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

/**
 * Saves a pending capture into IndexedDB for the audit page to consume.
 * @param {Object} captureData
 * @param {Blob} captureData.blob - JPEG Blob
 * @param {number} captureData.width - capture width
 * @param {number} captureData.height - capture height
 * @param {number} captureData.bytes - blob size in bytes
 * @param {'videoFrame'|'takePhoto'} captureData.method - capture method
 * @param {number} [captureData.createdAt] - epoch ms (defaults to Date.now())
 * @param {{conf: number, box: number[]}|null} [captureData.board] - board bounding box
 * @param {number|null} [captureData.exifOrientation] - EXIF orientation tag (T11)
 * @returns {Promise<void>}
 */
export async function savePendingCapture(captureData) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const record = {
            blob: captureData.blob,
            width: captureData.width,
            height: captureData.height,
            bytes: captureData.bytes ?? (captureData.blob ? captureData.blob.size : 0),
            method: captureData.method || 'videoFrame',
            createdAt: captureData.createdAt || Date.now(),
            board: captureData.board || null,
            exifOrientation: captureData.exifOrientation ?? null
        };
        const req = store.put(record, PENDING_KEY);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
        tx.oncomplete = () => db.close();
    });
}

/**
 * Retrieves the pending capture from IndexedDB.
 * Returns null if no record exists or if the record is expired (> 15 min old).
 * @param {boolean} [validateTTL=true]
 * @returns {Promise<Object|null>}
 */
export async function getPendingCapture(validateTTL = true) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(PENDING_KEY);
        req.onsuccess = () => {
            const record = req.result;
            if (!record) {
                resolve(null);
                return;
            }
            if (validateTTL && record.createdAt && (Date.now() - record.createdAt > MAX_AGE_MS)) {
                resolve(null);
                return;
            }
            resolve(record);
        };
        req.onerror = () => reject(req.error);
        tx.oncomplete = () => db.close();
    });
}

/**
 * Deletes the pending capture from IndexedDB (one-shot cleanup).
 * @returns {Promise<void>}
 */
export async function deletePendingCapture() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const req = store.delete(PENDING_KEY);
        req.onsuccess = () => resolve();
        req.onerror = () => reject(req.error);
        tx.oncomplete = () => db.close();
    });
}
