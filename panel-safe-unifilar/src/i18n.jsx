import React, { createContext, useContext, useState } from 'react';

// Shares the same localStorage key as the rest of the PanelSafe site
// (app/frontend/index.html, analysis.html, blog.html) so a language choice
// made anywhere on the site follows the user into this app too.
const STORAGE_KEY = 'breakerLang';

// SCOPE NOTE: this only translates the app's surrounding UI (buttons, labels,
// hints, editable-table column headers). It deliberately does NOT translate:
//   - the SVG diagram's own embedded text (IGA GEN, Sobretensiones, D.I.,
//     BAJO TUBO=, the CONSUMOS/Pcal/Un/Sf/Tubo/LONGITUD "cajetin" table) --
//     that diagram is the literal document an installer submits for REBT
//     compliance in Spain, and translating it would produce a document that
//     doesn't match what Spanish authorities actually expect.
//   - regional_fields.json (region names + region-specific bureaucratic
//     field labels like "Número de Registro Territorial") -- these are
//     official Spanish administrative paperwork terms with no translated
//     equivalent on the actual forms an installer files.
//   - rebt_defaults.json circuit names -- already a Spanish/English hybrid
//     and feeds directly into the exported diagram's usage column.
const translations = {
  en: {
    nav_home: 'Home', nav_analysis: 'AI Analyzer', nav_unifilar: 'Diagram Generator', nav_blog: 'Blog',
    brand_sub: 'Unifilar Engine',
    disclaimer: 'This tool is designed to help users prepare electrical circuit diagrams. It is not a substitute for professional technical knowledge: a certified electrician must review the tool before any official submission.',
    compliance_title: 'REBT-standard format',
    compliance_details: 'Draft for review — an authorized installer must validate it before official submission in {region}',
    footer: 'PanelSafe Unifilar v1.0.0 | Compliant with Spain\'s Low Voltage Electrotechnical Regulation (REBT) - ITC-BT-25 & UNE-EN 60617',

    region_label: 'Autonomous Community (Region):',
    region_placeholder: 'Select a region...',
    region_metadata_suffix: 'Required Regional Data',

    editor_title: 'Shorthand Code Editor',
    editor_help: 'Type board layouts using shorthand formatting or use the macros below to auto-fill common circuits.',
    macros_label: 'Add Circuits (REBT):',
    insert_main_line: '+ Main Line (IGA/SPD/RCD)',
    insert_main_title: 'Insert standard main breaker lines',
    insert_circuit_title: 'Insert standard',
    editor_placeholder: '// Write the diagram here:\n[IGA:40A:2P] -> [SPD] -> [RCD:40A:30mA:2P]\n[PIA:10A:1.5mm:T18] -> C1\n[PIA:16A:2.5mm:T20] -> C2',

    visualizer_title: 'Official Unifilar Diagram (UNE-EN 60617)',
    download_png: 'Download PNG',
    print_pdf: 'Print / Save PDF',
    empty_state: 'No breaker detected. Upload a photo in the Analysis tab or enter circuits in the editor on the left.',
    table_title: 'Electrical Panel Technical Data (Edit Values):',
    col_ref: 'Ref', col_usage: 'Usage / Load', col_power: 'Power (W)', col_voltage: 'Voltage (V)', col_cable: 'Cable (mm²)', col_tube: 'Tube (mm)',
    unreviewed_badge: 'Unreviewed',
    unreviewed_tooltip: 'Automatic suggestion, pending review',
    zoom_in: 'Zoom In', zoom_out: 'Zoom Out', zoom_reset: 'Reset',
    zoom_hint: 'Scroll to zoom · drag to pan · double-click to reset',

    usage_lighting: 'Lighting', usage_general_plugs: 'General Purpose Sockets', usage_bathroom_kitchen: 'Bathroom & Auxiliary Kitchen Sockets',
    usage_laundry: 'Washing Machine / Water Heater', usage_kitchen_oven: 'Kitchen & Oven', usage_ev_charger: 'Electric Vehicle Charger',
    usage_circuit_prefix: 'Circuit',
  },
  es: {
    nav_home: 'Inicio', nav_analysis: 'Analizador IA', nav_unifilar: 'Generador Unifilar', nav_blog: 'Blog',
    brand_sub: 'Unifilar Engine',
    disclaimer: 'Esta herramienta está diseñada para ayudar a los usuarios a preparar esquemas de circuitos eléctricos. No sustituye el conocimiento técnico profesional: un electricista certificado debe revisar la herramienta antes de realizar cualquier presentación oficial.',
    compliance_title: 'Formato según normativa REBT',
    compliance_details: 'Borrador para revisión — un instalador autorizado debe validarlo antes de su presentación oficial en {region}',
    footer: 'PanelSafe Unifilar v1.0.0 | Cumplimiento Reglamento Electrotécnico de Baja Tensión (REBT) - ITC-BT-25 & UNE-EN 60617',

    region_label: 'Comunidad Autónoma (Region):',
    region_placeholder: 'Seleccione una comunidad...',
    region_metadata_suffix: 'Datos Regionales Obligatorios',

    editor_title: 'Editor de Código Abreviado',
    editor_help: 'Escriba el esquema del cuadro usando el formato abreviado o use las macros de abajo para autocompletar circuitos comunes.',
    macros_label: 'Añadir Circuitos (REBT):',
    insert_main_line: '+ Línea Principal (IGA/SPD/RCD)',
    insert_main_title: 'Insertar líneas estándar de protección general',
    insert_circuit_title: 'Insertar estándar de',
    editor_placeholder: '// Escribe el esquema aquí:\n[IGA:40A:2P] -> [SPD] -> [RCD:40A:30mA:2P]\n[PIA:10A:1.5mm:T18] -> C1\n[PIA:16A:2.5mm:T20] -> C2',

    visualizer_title: 'Esquema Unifilar Oficial (UNE-EN 60617)',
    download_png: 'Descargar PNG',
    print_pdf: 'Imprimir / Guardar PDF',
    empty_state: 'Ningún interruptor detectado. Sube una foto en la pestaña de Análisis o introduce circuitos en el editor de la izquierda.',
    table_title: 'Datos Técnicos del Cuadro Eléctrico (Editar Valores):',
    col_ref: 'Ref', col_usage: 'Uso / Consumo', col_power: 'Potencia (W)', col_voltage: 'Tensión (V)', col_cable: 'Cable (mm²)', col_tube: 'Tubo (mm)',
    unreviewed_badge: 'Sin revisar',
    unreviewed_tooltip: 'Sugerencia automática, pendiente de revisión',
    zoom_in: 'Acercar', zoom_out: 'Alejar', zoom_reset: 'Restablecer',
    zoom_hint: 'Rueda para zoom · arrastra para mover · doble clic para restablecer',

    usage_lighting: 'Alumbrado', usage_general_plugs: 'Tomas de Uso General', usage_bathroom_kitchen: 'Tomas Baño y Cocina Auxiliar',
    usage_laundry: 'Lavadora / Termo', usage_kitchen_oven: 'Cocina y Horno', usage_ev_charger: 'Cargador Vehículo Eléctrico',
    usage_circuit_prefix: 'Circuito',
  },
  fr: {
    nav_home: 'Accueil', nav_analysis: 'Analyseur IA', nav_unifilar: 'Générateur de schéma', nav_blog: 'Blog',
    brand_sub: 'Unifilar Engine',
    disclaimer: "Cet outil est conçu pour aider les utilisateurs à préparer des schémas de circuits électriques. Il ne remplace pas les connaissances techniques professionnelles : un électricien certifié doit vérifier l'outil avant toute soumission officielle.",
    compliance_title: 'Format conforme à la norme REBT',
    compliance_details: "Brouillon à valider — un installateur agréé doit le valider avant sa présentation officielle en {region}",
    footer: "PanelSafe Unifilar v1.0.0 | Conforme au règlement électrotechnique basse tension espagnol (REBT) - ITC-BT-25 & UNE-EN 60617",

    region_label: 'Communauté autonome (région) :',
    region_placeholder: 'Sélectionnez une communauté...',
    region_metadata_suffix: 'Données régionales obligatoires',

    editor_title: 'Éditeur de code abrégé',
    editor_help: "Saisissez la disposition du tableau au format abrégé, ou utilisez les macros ci-dessous pour préremplir les circuits courants.",
    macros_label: 'Ajouter des circuits (REBT) :',
    insert_main_line: '+ Ligne principale (IGA/SPD/RCD)',
    insert_main_title: 'Insérer les lignes standard de protection générale',
    insert_circuit_title: 'Insérer le standard',
    editor_placeholder: '// Écrivez le schéma ici :\n[IGA:40A:2P] -> [SPD] -> [RCD:40A:30mA:2P]\n[PIA:10A:1.5mm:T18] -> C1\n[PIA:16A:2.5mm:T20] -> C2',

    visualizer_title: 'Schéma unifilaire officiel (UNE-EN 60617)',
    download_png: 'Télécharger PNG',
    print_pdf: 'Imprimer / Enregistrer en PDF',
    empty_state: "Aucun disjoncteur détecté. Téléversez une photo dans l'onglet Analyse ou saisissez les circuits dans l'éditeur à gauche.",
    table_title: 'Données techniques du tableau électrique (modifier les valeurs) :',
    col_ref: 'Réf', col_usage: 'Usage / Charge', col_power: 'Puissance (W)', col_voltage: 'Tension (V)', col_cable: 'Câble (mm²)', col_tube: 'Gaine (mm)',
    unreviewed_badge: 'Non vérifié',
    unreviewed_tooltip: 'Suggestion automatique, en attente de vérification',
    zoom_in: 'Zoomer', zoom_out: 'Dézoomer', zoom_reset: 'Réinitialiser',
    zoom_hint: 'Molette pour zoomer · glisser pour déplacer · double-clic pour réinitialiser',

    usage_lighting: 'Éclairage', usage_general_plugs: 'Prises usage général', usage_bathroom_kitchen: 'Prises salle de bain et cuisine auxiliaire',
    usage_laundry: 'Lave-linge / Chauffe-eau', usage_kitchen_oven: 'Cuisine et four', usage_ev_charger: 'Chargeur véhicule électrique',
    usage_circuit_prefix: 'Circuit',
  },
};

const LanguageContext = createContext(null);

export function LanguageProvider({ children }) {
  const [lang, setLangState] = useState(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return translations[saved] ? saved : 'es';
  });

  const setLang = (newLang) => {
    if (!translations[newLang]) return;
    setLangState(newLang);
    localStorage.setItem(STORAGE_KEY, newLang);
  };

  const t = (key, vars) => {
    let str = (translations[lang] && translations[lang][key]) || translations.en[key] || key;
    if (vars) {
      Object.entries(vars).forEach(([k, v]) => {
        str = str.replace(`{${k}}`, v);
      });
    }
    return str;
  };

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error('useLanguage must be used within a LanguageProvider');
  return ctx;
}
