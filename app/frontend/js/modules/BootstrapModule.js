/**
 * BootstrapModule (Deep Module)
 * Handles the 'Basement Problem' by verifying the presence of local assets
 * in the browser's CacheStorage before permitting entry to the DetectionModule.
 */
export const STATUS = { 
    COLD: 'cold',    // Assets not in cache
    WARM: 'warm',    // Assets fully cached
    ERROR: 'error'   // Cache API inaccessible
};

export class BootstrapModule {
    #assets = ['yolo26-nano.onnx', 'config.json'];
    #cacheName = 'panelsafe-v2';

    /**
     * Checks if all required model assets are stored in the persistent cache.
     * @returns {Promise<string>} Current cache status from STATUS enum.
     */
    async checkCacheState() {
        if (!('caches' in window)) {
            console.error("Cache API not supported");
            return STATUS.ERROR;
        }

        try {
            const cache = await window.caches.open(this.#cacheName);
            
            // Check all assets in parallel for maximum speed during splash screen
            const checks = await Promise.all(
                this.#assets.map(asset => cache.match(asset))
            );

            // If every asset check returned a valid Response object, we are WARM
            const allFound = checks.every(response => response !== undefined);
            
            return allFound ? STATUS.WARM : STATUS.COLD;
        } catch (error) {
            console.error("Bootstrap Cache check failed:", error);
            return STATUS.ERROR;
        }
    }

    /**
     * Manually triggers a prefetch of assets. 
     * In production, this would typically be handled by the Service Worker,
     * but this provides a fallback UI-triggered mechanism.
     */
    async prefetchAssets() {
        try {
            const cache = await window.caches.open(this.#cacheName);
            await cache.addAll(this.#assets);
            return STATUS.WARM;
        } catch (error) {
            console.error("Prefetch failed:", error);
            return STATUS.ERROR;
        }
    }
}
