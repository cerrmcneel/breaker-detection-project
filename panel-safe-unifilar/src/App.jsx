import React, { useState, useMemo } from 'react';
import RegionSelector from './components/RegionSelector';
import Editor from './components/Editor';
import BoardVisualizer from './components/BoardVisualizer';
import { parseUnifilarText } from './parser/diagramParser';
import { useLanguage } from './i18n.jsx';
import './App.css';

function App() {
  const { lang, setLang, t } = useLanguage();

  // 1. STATE MANAGEMENT
  // textCode holds the raw shorthand text input from the editor.
  const [textCode, setTextCode] = useState(() => {
    return localStorage.getItem("panelsafe_unifilar_code") || (
      `// PanelSafe Shorthand Code - Esquema Unifilar\n` +
      `// Formato: [TIPO:RATING:CABLE:TUBO] -> Destino\n\n` +
      `[IGA:40A:2P] -> [SPD] -> [RCD:40A:30mA:2P]\n\n` +
      `// Circuitos de protección individual (REBT)\n` +
      `[PIA:10A:1.5mm:T18] -> C1\n` +
      `[PIA:16A:2.5mm:T20] -> C2\n` +
      `[PIA:25A:6mm:T25] -> C3\n` +
      `[PIA:20A:4mm:T20] -> C4\n` +
      `[PIA:16A:2.5mm:T20] -> C5\n` +
      `[PIA:16A:2.5mm:T25] -> C13`
    );
  });

  // selectedRegion holds the current Comunidad Autónoma select value.
  const [selectedRegion, setSelectedRegion] = useState('Comunidad Valenciana');

  // metadataValues holds a key-value dictionary of region-specific fields.
  const [metadataValues, setMetadataValues] = useState({
    reg_number: 'CV-R-87421',
    installer_company: 'ElectroMarjal S.L.',
    icc: '6 kA',
    voltage_drop: '1.5%'
  });

  // 2. DATA FLOW & COMPUTED STATE (REAL-TIME PARSING)
  // useMemo runs the parsing logic whenever textCode changes.
  // This computes a flat array of nodes with targets to represent the board hierarchy.
  const parsedNodes = useMemo(() => {
    return parseUnifilarText(textCode);
  }, [textCode]);

  // 3. EVENT HANDLERS
  const handleTextChange = (newValue) => {
    setTextCode(newValue);
  };

  const handleRegionChange = (newRegion) => {
    setSelectedRegion(newRegion);
    // Values are keyed per-field, and each region only renders its own field
    // keys (see regional_fields.json), so leftover values from a previous
    // region are simply not shown -- no need to wipe them, and doing so
    // silently deleted anything typed if the dropdown was touched again.
  };

  const handleMetadataChange = (key, value) => {
    setMetadataValues((prev) => ({
      ...prev,
      [key]: value
    }));
  };

  return (
    <div className="app-shell">
      {/* Premium Header */}
      <header className="app-header">
        <div className="header-logo-container">
          <a href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <svg viewBox="0 0 24 24" width="32" height="32" className="header-logo-icon">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <div className="logo-text">
              <span className="brand-name">PanelSafe</span>
              <span className="brand-sub">{t('brand_sub')}</span>
            </div>
          </a>
        </div>
        <div className="nav-links">
          <a href="/" className="nav-link">{t('nav_home')}</a>
          <a href="/analysis.html" className="nav-link">{t('nav_analysis')}</a>
          <a href="/unifilar/" className="nav-link active">{t('nav_unifilar')}</a>
          <a href="/blog.html" className="nav-link">{t('nav_blog')}</a>
        </div>
        <div className="lang-switcher-nav" style={{ display: 'flex', gap: '6px', marginLeft: '12px' }}>
          {['en', 'es', 'fr'].map((code) => (
            <button
              key={code}
              type="button"
              onClick={() => setLang(code)}
              className={`lang-btn-nav${lang === code ? ' active' : ''}`}
              style={{
                padding: '6px 12px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 700,
                border: '1px solid rgba(145, 55, 175, 0.3)', cursor: 'pointer',
                background: lang === code ? 'var(--primary-light, #9137af)' : 'transparent',
                color: lang === code ? '#fff' : 'inherit',
              }}
            >
              {code.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      {/* Persistent disclaimer -- not tied to region selection, always visible */}
      <div className="disclaimer-banner">
        {t('disclaimer')}
      </div>

      {/* Main Grid Layout */}
      <main className="app-main-grid">
        {/* Left Control Panel: Inputs and Configurations */}
        <section className="control-panel">
          <div className="panel-card region-card">
            <RegionSelector
              selectedRegion={selectedRegion}
              onRegionChange={handleRegionChange}
              metadataValues={metadataValues}
              onMetadataChange={handleMetadataChange}
            />
          </div>

          <div className="panel-card editor-card-wrapper">
            <Editor
              textCode={textCode}
              onTextChange={handleTextChange}
            />
          </div>
        </section>

        {/* Right Output Panel: Visualized Schematic Canvas */}
        <section className="visual-canvas-panel">
          <div className="panel-card visualizer-card">
            <BoardVisualizer parsedNodes={parsedNodes} />
            
            {/* Live Compliance & Metadata Overlay */}
            {selectedRegion && (
              <div className="canvas-footer-summary">
                <div className="compliance-stamp">
                  <div className="stamp-icon">i</div>
                  <div>
                    <div className="stamp-title">{t('compliance_title')}</div>
                    <div className="stamp-details">{t('compliance_details', { region: selectedRegion })}</div>
                  </div>
                </div>
                
                {Object.keys(metadataValues).length > 0 && (
                  <div className="metadata-summary-chips">
                    {Object.entries(metadataValues).map(([key, val]) => (
                      val && (
                        <div key={key} className="meta-chip">
                          <span className="chip-key">{key.toUpperCase()}:</span>
                          <span className="chip-val">{val}</span>
                        </div>
                      )
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Technical Footer */}
      <footer className="app-footer">
        <p>{t('footer')}</p>
      </footer>
    </div>
  );
}

export default App;
