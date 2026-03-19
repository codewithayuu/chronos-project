import React, { useEffect, useRef, useState, useCallback } from 'react';

function AnimatedNumber({
  value,
  decimals = 2,
  duration = 600,
  className = '',
  style = {},
  prefix = '',
  suffix = '',
}) {
  const [displayValue, setDisplayValue] = useState(value);
  const animationRef = useRef(null);
  const startValueRef = useRef(value);
  const startTimeRef = useRef(null);

  const animate = useCallback(
    (timestamp) => {
      if (!startTimeRef.current) startTimeRef.current = timestamp;
      const elapsed = timestamp - startTimeRef.current;
      const progress = Math.min(elapsed / duration, 1);

      // Expo ease out
      const eased = 1 - Math.pow(1 - progress, 3);
      const current =
        startValueRef.current + (value - startValueRef.current) * eased;

      setDisplayValue(current);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    },
    [value, duration]
  );

  useEffect(() => {
    if (value == null) {
      setDisplayValue(null);
      return;
    }

    if (displayValue == null) {
      setDisplayValue(value);
      startValueRef.current = value;
      return;
    }

    startValueRef.current = displayValue;
    startTimeRef.current = null;

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, animate]);

  if (displayValue == null) {
    return (
      <span className={className} style={style}>
        {prefix}--{suffix}
      </span>
    );
  }

  return (
    <span className={className} style={style}>
      {prefix}
      {displayValue.toFixed(decimals)}
      {suffix}
    </span>
  );
}

export default React.memo(AnimatedNumber);
