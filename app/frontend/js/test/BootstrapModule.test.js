/**
 * TDD Test Suite for BootstrapModule
 * Verifies the integrity of the offline baseline detection.
 */
import { BootstrapModule, STATUS } from '../modules/BootstrapModule.js';

async function runBootstrapTests() {
    console.group("🚦 BootstrapModule: Unit Tests");
    const bootstrap = new BootstrapModule();

    // Test 1: Cold Cache Detection
    // Note: We intentionally clear the cache to verify the COLD state
    try {
        await window.caches.delete('panelsafe-v2');
        const state = await bootstrap.checkCacheState();
        console.assert(state === STATUS.COLD, `Test 1 Failed: Expected COLD, got ${state}`);
        if(state === STATUS.COLD) console.log("✅ Test 1: Cold Cache Detection Passed");
    } catch (e) {
        console.error("❌ Test 1 Error:", e);
    }

    // Test 2: Warm Cache Detection
    try {
        // Manually "Warm" the cache
        await bootstrap.prefetchAssets(); 
        const state = await bootstrap.checkCacheState();
        console.assert(state === STATUS.WARM, `Test 2 Failed: Expected WARM, got ${state}`);
        if(state === STATUS.WARM) console.log("✅ Test 2: Warm Cache Detection Passed");
    } catch (e) {
        console.error("❌ Test 2 Error:", e);
    }

    console.groupEnd();
}

// Export for manual trigger or auto-run
if (window.location.search.includes('test=true')) {
    runBootstrapTests();
}
