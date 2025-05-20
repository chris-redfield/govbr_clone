
/* - ++resource++lgpd.plugin/block_cookies.js - */
// https://www.gov.br/portal_javascripts/++resource++lgpd.plugin/block_cookies.js?original=1
const cookieexpires=365
const cookiepolicy='Strict'
function getCookie(name){var value="; "+document.cookie;var parts=value.split("; "+name+"=");return parts.length<2?undefined:parts.pop().split(";").shift()}
function setCookie(name,value,expiryDays,domain,path,secure,samesite){var exdate=new Date();exdate.setHours(exdate.getHours()+(expiryDays||cookieexpires) * 24);allCookies=document.cookie;var cookie=[name+"="+value,"expires="+exdate.toUTCString(),"path="+(path||"/"),];if(domain){cookie.push("domain="+domain)}
if(secure){cookie.push("secure")}
if(samesite){cookie.push("SameSite="+samesite)}
document.cookie=cookie.join(";")}
function getHabilitados(cookiename){var listaHabilitados=[];var lista=[];cookieLgpd=getCookie(cookiename);if(cookieLgpd){cookieDataLgpd=JSON.parse(cookieLgpd);cookieGroup=cookieDataLgpd.cookieGroups;cookieGroup.forEach((grupo)=>{if(grupo.groupSelected==true){listaHabilitados.push(grupo.groupId.toString())}
grupo.cookieList.forEach((b)=>{lista.push(b.cookieId,b.cookieName,b.cookieSelected);if(b.cookieSelected==true){listaHabilitados.push(b.cookieId.toString())}})})}
return listaHabilitados}
function gravarCookie(json){var cookiebarSaida=json;cookieData=JSON.parse(cookiebarSaida);setCookie(cookiename,JSON.stringify(cookieData),cookieexpires,'','/',false,cookiepolicy)}

