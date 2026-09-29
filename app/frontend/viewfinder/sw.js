// PanelSafe Live Viewfinder Service Worker (T9 Offline "Basement" Mode)
// Cache Name is explicitly versioned so new weights/assets supersede old versions.
// Bump it whenever the model, runtime or page changes.
const CACHE_NAME = "panelsafe-viewfinder-prod-2026-09-30";

const PRECACHE_URLS = [
  "./",
  "./index.html",
  "./vendor/ort.min.js",
  "./vendor/ort-wasm-simd-threaded.wasm",
  "./vendor/ort-wasm-simd-threaded.mjs",
  "./board_viewfinder.onnx",
  "/js/modules/captureHandoff.js"
];
// samples/ is deliberately absent: those panels are not published on the public
// site, so precaching them would 404 and keep the HUD from ever reporting READY.

// --- Install Phase: Precache all critical assets ---
self.addEventListener("install", (event) => {
  console.log(`[ServiceWorker ${CACHE_NAME}] Installing and precaching offline assets...`);
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      for (const url of PRECACHE_URLS) {
        try {
          await cache.add(url);
          console.log(`[ServiceWorker] Precached: ${url}`);
        } catch (err) {
          console.error(`[ServiceWorker] Precache failed for ${url}:`, err);
        }
      }
    }).then(() => self.skipWaiting())
  );
});

// --- Activate Phase: Purge obsolete caches from previous versions ---
self.addEventListener("activate", (event) => {
  console.log(`[ServiceWorker ${CACHE_NAME}] Activating and cleaning stale caches...`);
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            console.log(`[ServiceWorker] Deleting obsolete cache: ${name}`);
            return caches.delete(name);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// --- Fetch Phase: Cache-First strategy to ensure 100% offline functionality ---
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Bypass API calls (benchmarks, telemetry) and non-GET requests
  if (event.request.method !== "GET" || url.pathname.startsWith("/api/")) {
    return;
  }

  // Page navigations are network-first, so a deployed change reaches phones on the
  // next online load instead of being pinned by the cache forever. Offline (or on
  // a network that hangs for 3 s), the cached page is served.
  if (event.request.mode === "navigate") {
    event.respondWith(networkFirstPage(event.request));
    return;
  }

  // Everything else (runtime, WASM, model) stays cache-first: large and versioned
  // through CACHE_NAME.
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }

      // If not in cache, fetch from network and dynamically cache
      return fetch(event.request).then((networkResponse) => {
        if (networkResponse && networkResponse.status === 200) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      }).catch((err) => {
        console.warn(`[ServiceWorker] Network request failed for: ${event.request.url}`, err);
        throw err;
      });
    })
  );
});

async function networkFirstPage(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const timeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("network timeout")), 3000));
    const response = await Promise.race([fetch(request), timeout]);
    if (response && response.status === 200) {
      cache.put("./", response.clone());
    }
    return response;
  } catch (err) {
    console.warn("[ServiceWorker] Page fetch failed, serving cached copy:", err);
    const cached = (await cache.match(request, { ignoreSearch: true }))
      || (await cache.match("./"))
      || (await cache.match("./index.html"));
    if (cached) return cached;
    throw err;
  }
}

// --- Client Communication: Report cache readiness to HUD ---
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "CHECK_OFFLINE_READY") {
    caches.open(CACHE_NAME).then(async (cache) => {
      const keys = await cache.keys();
      const cachedUrls = keys.map((k) => k.url);
      const isComplete = PRECACHE_URLS.every((u) => {
        return cachedUrls.some((cu) => cu.endsWith(u.replace("./", "")));
      });

      if (event.ports && event.ports[0]) {
        event.ports[0].postMessage({
          ready: isComplete,
          cachedCount: keys.length,
          cacheName: CACHE_NAME
        });
      }
    });
  }
});
