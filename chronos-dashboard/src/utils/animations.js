// =============================================
// ANIMATION VARIANTS & SPRING CONFIGS
// Premium motion choreography constants.
// Never use linear or ease-in-out — always
// spring physics or expo curves.
// =============================================

// Spring physics presets
export const SPRING_SNAPPY = {
  type: 'spring',
  stiffness: 400,
  damping: 30,
  mass: 0.8,
};

export const SPRING_SMOOTH = {
  type: 'spring',
  stiffness: 100,
  damping: 20,
  mass: 1,
};

export const SPRING_GENTLE = {
  type: 'spring',
  stiffness: 60,
  damping: 15,
  mass: 1.2,
};

export const SPRING_BOUNCY = {
  type: 'spring',
  stiffness: 300,
  damping: 24,
  mass: 0.6,
};

// Expo ease for CSS-driven transitions
export const EASE_OUT_EXPO = [0.16, 1, 0.3, 1];

// ---- Page Transition Variants ----
export const pageVariants = {
  initial: {
    opacity: 0,
    y: 16,
    filter: 'blur(6px)',
  },
  animate: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: {
      duration: 0.5,
      ease: EASE_OUT_EXPO,
      staggerChildren: 0.06,
      delayChildren: 0.1,
    },
  },
  exit: {
    opacity: 0,
    y: -10,
    filter: 'blur(4px)',
    transition: {
      duration: 0.3,
      ease: EASE_OUT_EXPO,
    },
  },
};

// ---- Ward Grid Variants ----
export const wardGridVariants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.08,
    },
  },
};

export const wardCardVariants = {
  initial: {
    opacity: 0,
    y: 20,
    scale: 0.97,
  },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      ...SPRING_SMOOTH,
    },
  },
};

// ---- Detail View Panel Stagger ----
export const detailColumnVariants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.15,
    },
  },
};

export const detailPanelVariants = {
  initial: {
    opacity: 0,
    y: 18,
    scale: 0.985,
  },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      ...SPRING_SMOOTH,
    },
  },
};

// ---- Chart Stagger ----
export const chartStackVariants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.06,
      delayChildren: 0.1,
    },
  },
};

export const chartItemVariants = {
  initial: {
    opacity: 0,
    x: -12,
    scale: 0.98,
  },
  animate: {
    opacity: 1,
    x: 0,
    scale: 1,
    transition: {
      ...SPRING_SMOOTH,
    },
  },
};

// ---- Alert Feed Variants ----
export const alertFeedOverlayVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.25, ease: EASE_OUT_EXPO } },
  exit: { opacity: 0, transition: { duration: 0.2, ease: EASE_OUT_EXPO } },
};

export const alertFeedPanelVariants = {
  initial: { x: '100%', opacity: 0 },
  animate: {
    x: 0,
    opacity: 1,
    transition: {
      ...SPRING_SNAPPY,
      staggerChildren: 0.04,
      delayChildren: 0.15,
    },
  },
  exit: {
    x: '100%',
    opacity: 0,
    transition: { duration: 0.3, ease: EASE_OUT_EXPO },
  },
};

export const alertItemVariants = {
  initial: { opacity: 0, x: 20, scale: 0.96 },
  animate: {
    opacity: 1,
    x: 0,
    scale: 1,
    transition: { ...SPRING_SMOOTH },
  },
  exit: {
    opacity: 0,
    x: 20,
    scale: 0.96,
    transition: { duration: 0.2, ease: EASE_OUT_EXPO },
  },
};

// ---- Intervention Card Stagger ----
export const interventionListVariants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.07,
      delayChildren: 0.1,
    },
  },
};

export const interventionItemVariants = {
  initial: { opacity: 0, y: 12, scale: 0.97 },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { ...SPRING_SMOOTH },
  },
};

// ---- Tooltip / Popover ----
export const tooltipVariants = {
  initial: { opacity: 0, y: 4, scale: 0.95 },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { ...SPRING_SNAPPY },
  },
  exit: {
    opacity: 0,
    y: 4,
    scale: 0.95,
    transition: { duration: 0.15 },
  },
};

// ---- Gauge / Number Counter ----
export const gaugeRevealVariants = {
  initial: { opacity: 0, scale: 0.9 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: {
      ...SPRING_GENTLE,
      delay: 0.2,
    },
  },
};

// ---- Severity Pulse Keyframes (for inline motion) ----
export const severityPulse = {
  CRITICAL: {
    scale: [1, 1.01, 1],
    transition: { duration: 1, repeat: Infinity, ease: 'easeInOut' },
  },
  WARNING: {
    scale: [1, 1.005, 1],
    transition: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
  },
  WATCH: {
    scale: [1, 1.003, 1],
    transition: { duration: 3, repeat: Infinity, ease: 'easeInOut' },
  },
  NONE: {},
};
