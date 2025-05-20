
/* - ++resource++govbr.policy/application_cookiebar_1_3_55.js - */
// https://www.gov.br/portal_javascripts/++resource++govbr.policy/application_cookiebar_1_3_55.js?original=1
var habilitar=false
var idioma=''
var mensagens=''
var cookiename='lgpd-cookie-v2'
var config_url=urlConfiguracao()
function urlConfiguracao(){getUrl=window.location;portal_url=window.portal_url
if(portal_url){baseUrl=portal_url} else{baseUrl=getUrl.protocol+"//"+getUrl.host}
url_array=getUrl.pathname.split('/');if(url_array[1]=='pt-br'||url_array[1]=='en'||url_array[1]=='es'){baseUrl=baseUrl+'/'+url_array[1]}
if(url_array[2]=='pt-br'||url_array[2]=='en'||url_array[2]=='es'){if(portal_url){baseUrl=baseUrl+'/'+url_array[2]} else{baseUrl=baseUrl+'/'+url_array[1]+'/'+url_array[2]}}
baseUrl=baseUrl+'/@@configuracoes_view';return baseUrl}
function hide(elements){elements=elements.length?elements:[elements];for(var index=0;index<elements.length;index++){elements[index].style.display='none'}}
function callback(json){gravarCookie(json);_createEventIcone();showTooltip();showBand()}
function click_redefinir(event){event.stopPropagation()
let divRedefinirCookies=document.querySelector('.redefinir-cookies');if(event.keyCode===32||event.keyCode===13){divRedefinirCookies.click()}}
function _createEventIcone(){let divRedefinirCookies=document.querySelector('.redefinir-cookies')}
function _redefinirCookies(){fetch(config_url).then(res=>res.json()).then((out)=>{habilitar=out['habilitar'];idioma=out['idioma'];mensagens=out['mensagens'];cookiename=out['cookie_name'];if(habilitar){const defaultInfoList=JSON.parse(mensagens)
const outputInfo=JSON.parse(getCookie(cookiename))
defaultInfoList.forEach(defaultInfo=>{defaultInfo.selectAll=outputInfo.selectAll
defaultInfo.cookieGroups.forEach((group,groupIndex)=>{try{group.groupSelected=outputInfo.g[groupIndex].on} catch(error){}
group.cookieList.forEach((cookie,cookieIndex)=>{try{cookie.cookieSelected=outputInfo.g[groupIndex].on} catch(error){}})})})
let divRedefinirCookies=document.querySelector('.redefinir-cookies')
divRedefinirCookies.style.display="none"
novaCookieBar(idioma,JSON.stringify(defaultInfoList),'open');if(idioma=='en'){var label='Save'} else if(idioma=='es'){var label='Guardar'} else{var label='Salvar'}
document.querySelectorAll('.br-button.btn-accept').forEach((button)=>{button.innerHTML=label;button.setAttribute('aria-label',label)})}}).catch(err=>console.error(err))}
function novaCookieBar(idioma,mensagens,modo){const cookiebarList=[]
for(const brCookiebar of window.document.querySelectorAll('.br-cookiebar')){cookiebarList.push(new BRCookiebar('br-cookiebar',brCookiebar,modo,mensagens,idioma,callback,))}}
