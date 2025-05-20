
/* - ++resource++lgpd.plugin/changeCookieImage.js - */
// https://www.gov.br/portal_javascripts/++resource++lgpd.plugin/changeCookieImage.js?original=1
function showTooltip(){if(window.matchMedia("(max-width: 766px)").matches){document.querySelectorAll('.cookie-tooltip').forEach(tooltip=>{tooltip.style.display='block'})
setTimeout(()=>{document.querySelectorAll('.cookie-tooltip').forEach(tooltip=>{tooltip.style.display='none'})},5000)}}
function showBand(){if(window.matchMedia("(min-width: 767px)").matches){document.querySelectorAll('.redefinir-cookies span').forEach(redefinirLabel=>{redefinirLabel.style['margin-left']='0'})}
setTimeout(()=>{document.querySelectorAll('.redefinir-cookies span').forEach(redefinirLabel=>{redefinirLabel.style.removeProperty('margin-left')})},5000)}
