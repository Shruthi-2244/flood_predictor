/* ==========================================================================
   AQUAGUARD AI - CLIENT JAVASCRIPT CONTROLLER
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initAboutTabs();
    initPredictionForm();
});

/* ==========================================================================
   DARK/LIGHT THEME CONTROLLER
   ========================================================================== */

function initTheme() {
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (!themeToggleBtn) return;

    // Check saved theme or system preference
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && systemPrefersDark)) {
        document.body.classList.add('dark-theme');
        updateThemeIcon(true);
    } else {
        document.body.classList.remove('dark-theme');
        updateThemeIcon(false);
    }

    themeToggleBtn.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark-theme');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        updateThemeIcon(isDark);
    });
}

function updateThemeIcon(isDark) {
    const icon = document.getElementById('theme-icon');
    if (!icon) return;
    
    if (isDark) {
        // Render Sun Icon
        icon.innerHTML = `
            <circle cx="12" cy="12" r="4"/>
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
        `;
    } else {
        // Render Moon Icon
        icon.innerHTML = `
            <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>
        `;
    }
}

/* ==========================================================================
   FORM CONTROLLER & VALIDATIONS
   ========================================================================== */

function initPredictionForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    const monthlyInputs = document.querySelectorAll('.rainfall-input');
    const annualField = document.getElementById('annual_rainfall');
    const visibilitySlider = document.getElementById('visibility');
    const visibilityNum = document.getElementById('visibility_num');
    const loadingOverlay = document.getElementById('loading-overlay');

    // 1. Sync Cloud Visibility Slider and Numeric Box
    if (visibilitySlider && visibilityNum) {
        // Slider moves -> update number box
        visibilitySlider.addEventListener('input', (e) => {
            visibilityNum.value = e.target.value;
        });

        // Number input changes -> update slider
        visibilityNum.addEventListener('input', (e) => {
            let val = parseInt(e.target.value);
            if (isNaN(val)) val = 0;
            if (val < 0) val = 0;
            if (val > 100) val = 100;
            
            e.target.value = val;
            visibilitySlider.value = val;
        });
    }

    // 2. Compute Annual Rainfall Summation in Real-time
    function calculateAnnualRainfall() {
        let total = 0;
        monthlyInputs.forEach(input => {
            const val = parseFloat(input.value);
            if (!isNaN(val) && val >= 0) {
                total += val;
            }
        });
        if (annualField) {
            annualField.value = total.toFixed(2);
        }
    }

    // Attach listener to all monthly inputs
    monthlyInputs.forEach(input => {
        input.addEventListener('input', () => {
            // Client-side negative prevention
            if (parseFloat(input.value) < 0) {
                input.value = 0;
            }
            calculateAnnualRainfall();
        });
    });

    // Run initial summation on page load
    calculateAnnualRainfall();

    // 3. Form Submission Overlay & Client Validation
    form.addEventListener('submit', (e) => {
        let isValid = true;
        let validationError = "";

        // Double check inputs are valid
        monthlyInputs.forEach(input => {
            const val = parseFloat(input.value);
            if (isNaN(val) || val < 0) {
                isValid = false;
                validationError = "All monthly rainfall values must be non-negative numeric parameters.";
                input.classList.add('error');
            } else {
                input.classList.remove('error');
            }
        });

        const visVal = parseFloat(visibilityNum.value);
        if (isNaN(visVal) || visVal < 0 || visVal > 100) {
            isValid = false;
            validationError = "Cloud visibility must be between 0% and 100%.";
        }

        if (!isValid) {
            e.preventDefault();
            alert(validationError);
            return;
        }

        // Show premium loading spinner during submission
        if (loadingOverlay) {
            loadingOverlay.classList.remove('hidden');
        }
    });

    // Reset handler to recalculate
    form.addEventListener('reset', () => {
        setTimeout(() => {
            calculateAnnualRainfall();
            if (visibilitySlider && visibilityNum) {
                visibilityNum.value = visibilitySlider.value;
            }
        }, 50); // slight delay to allow inputs to reset to default
    });
}

/* ==========================================================================
   ABOUT PAGE TABS DASHBOARD
   ========================================================================== */

function initAboutTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    if (tabBtns.length === 0) return;

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Remove active states
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            // Add active state to current elements
            btn.classList.add('active');
            const targetPane = document.getElementById(targetTab);
            if (targetPane) {
                targetPane.classList.add('active');
            }
        });
    });
}
