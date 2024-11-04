import React from 'react';

// ==============================|| LOGO SVG ||============================== //

const Logo = () => {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <svg width="300" height="100" viewBox="0 0 600 120" fill="none" xmlns="http://www.w3.org/2000/svg">
        {/* Left-aligned geometric symbol */}
        <g transform="translate(40, 25)">
          {/* Main hexagon with premium line weight */}
          <path 
            d="M30 0 L60 17 L60 52 L30 69 L0 52 L0 17 Z" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="1.5"
          />
          {/* Inner lines suggesting connectivity */}
          <path 
            d="M30 0 L30 69 M0 17 L60 17 M0 52 L60 52" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="1"
          />
          {/* Elegant dots at intersections */}
          <circle cx="30" cy="0" r="2" fill="currentColor"/>
          <circle cx="30" cy="69" r="2" fill="currentColor"/>
          <circle cx="0" cy="17" r="2" fill="currentColor"/>
          <circle cx="60" cy="17" r="2" fill="currentColor"/>
          <circle cx="0" cy="52" r="2" fill="currentColor"/>
          <circle cx="60" cy="52" r="2" fill="currentColor"/>
        </g>
        
        {/* Text Group */}
        <g transform="translate(120, 75)">
          {/* Text "FINSIGHT" */}
          <text 
            fontFamily="Arial, sans-serif" 
            fontWeight="300" 
            fontSize="42" 
            fill="currentColor"
            letterSpacing="2"
          >
            FINSIGHT
          </text>
          
          {/* Text "AI" with increased spacing */}
          <text 
            x="225" 
            fontFamily="Arial, sans-serif" 
            fontWeight="300" 
            fontSize="42" 
            fill="currentColor"
            letterSpacing="2"
          >
            AI
          </text>
        </g>
      </svg>
    </div>
  );
};

export default Logo;