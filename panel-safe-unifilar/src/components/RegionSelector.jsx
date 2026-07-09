import React from 'react';
import regionalFields from '../config/regional_fields.json';
import { useLanguage } from '../i18n.jsx';

// NOTE: region names and their per-region field labels (e.g. "Número de
// Registro Territorial") are deliberately left in Spanish -- these are
// official Spanish administrative paperwork terms with no translated
// equivalent on the actual forms an installer files. See i18n.js's top
// comment for the full scoping rationale.
export default function RegionSelector({
  selectedRegion,
  onRegionChange,
  metadataValues,
  onMetadataChange
}) {
  const { t } = useLanguage();
  const regions = Object.keys(regionalFields);

  return (
    <div className="region-selector-container">
      <div className="form-group">
        <label htmlFor="region-select" className="form-label">{t('region_label')}</label>
        <select
          id="region-select"
          value={selectedRegion}
          onChange={(e) => onRegionChange(e.target.value)}
          className="region-dropdown"
        >
          <option value="">{t('region_placeholder')}</option>
          {regions.map(r => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
      </div>

      {selectedRegion && regionalFields[selectedRegion] && (
        <div className="regional-metadata-fields">
          <h3 className="metadata-title">{selectedRegion} - {t('region_metadata_suffix')}</h3>
          <div className="fields-grid">
            {regionalFields[selectedRegion].map((field) => (
              <div className="form-group" key={field.key}>
                <label htmlFor={field.key} className="form-label">{field.label}:</label>
                <input
                  id={field.key}
                  type={field.type}
                  value={metadataValues[field.key] || ''}
                  placeholder={field.placeholder}
                  onChange={(e) => onMetadataChange(field.key, e.target.value)}
                  className="metadata-input"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
