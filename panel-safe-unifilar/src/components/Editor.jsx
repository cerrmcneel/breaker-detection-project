import React from 'react';
import rebtDefaults from '../config/rebt_defaults.json';
import { useLanguage } from '../i18n.jsx';

export default function Editor({ textCode, onTextChange }) {
  const { t } = useLanguage();

  // Inject standard circuit shorthand line
  const insertMacro = (circuitKey) => {
    const data = rebtDefaults[circuitKey];
    if (!data) return;

    const macroLine = `[PIA:${data.mcb}:${data.cable}:${data.tube}] -> ${circuitKey}\n`;
    onTextChange(textCode + (textCode.endsWith('\n') || textCode === '' ? '' : '\n') + macroLine);
  };

  // Inject general protection main line macro
  const insertMainHead = () => {
    const mainLine = `[IGA:40A:2P] -> [SPD] -> [RCD:40A:30mA:2P]\n`;
    onTextChange(textCode + (textCode.endsWith('\n') || textCode === '' ? '' : '\n') + mainLine);
  };

  return (
    <div className="editor-container">
      <h3 className="editor-title">{t('editor_title')}</h3>
      <p className="editor-help-text">
        {t('editor_help')}
      </p>

      {/* Quick Autofill Buttons */}
      <div className="macros-section">
        <span className="macro-label">{t('macros_label')}</span>
        <div className="macro-buttons">
          <button
            type="button"
            onClick={insertMainHead}
            className="macro-btn main-head-btn"
            title={t('insert_main_title')}
          >
            {t('insert_main_line')}
          </button>
          {Object.keys(rebtDefaults).map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => insertMacro(key)}
              className="macro-btn circuit-btn"
              title={`${t('insert_circuit_title')} ${rebtDefaults[key].name}`}
            >
              + {key} ({rebtDefaults[key].mcb})
            </button>
          ))}
        </div>
      </div>

      {/* Editor Shorthand Input */}
      <textarea
        value={textCode}
        onChange={(e) => onTextChange(e.target.value)}
        placeholder={t('editor_placeholder')}
        className="editor-textarea"
        rows={12}
      />
    </div>
  );
}
