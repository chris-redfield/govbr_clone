
/* - ++resource++govbr.policy/cookiebar_1_3_55.js - */
// https://www.gov.br/portal_javascripts/++resource++govbr.policy/cookiebar_1_3_55.js?original=1
const version=7
class BRCookiebar{constructor(name,component,mode,data,lang,callback){this.name=name
this.component=component
this.mode=mode?mode:'default'
this.data=this._setData(data,lang)
this.callback=callback
this._setUp()
this._setBehavior()}
_setData(data,lang){if(data){const dataObj=JSON.parse(data).filter((obj)=>{return obj.lang===lang})
if(dataObj.length>0){return dataObj[0]} else{return JSON.parse(data)[0]}} else{return null}}
_setUp(){this._showCookiebar()
this._setInitialData()
this._setSecondaryButton()
this._setPrimaryButton()
this._setInfoText()
this._setMainTitle()
this._setCloseButton()
this._setLinks()
this._setListHeader()
this._setEntryText()
this._setGroupInfo()
this._setNotification()}
_setBehavior(){this._setCheckAll()
this._setCheckGroup()
this._setCheckCookie()
this._setAcceptBehavior()
this._setRejectBehavior()
this._setResize()}
_showCookiebar(){this.component.classList.remove('d-none')
this.component.focus()
if(this.mode==='open'){this.component.classList.remove('default')
this.component.querySelectorAll('.actions .br-button.btn-manage').forEach((button)=>{button.classList.add('d-none')})
document.body.style.overflowY='hidden'
this._setOpenView()}
this._setRejectButton()}
_setInitialData(){if(!this.data.allOptOut){this.data.selectAll=true
this.data.allIndeterminate=false
this.data.cookieGroups.forEach((group)=>{group.groupOptOut=false
group.groupSelected=true
group.groupIndeterminate=false
group.cookieList.forEach((cookie)=>{cookie.cookieOptOut=false
cookie.cookieSelected=true})})} else if(this.data.selectAll===true){this.data.allIndeterminate=false
this.data.cookieGroups.forEach((group)=>{group.groupSelected=true
group.groupIndeterminate=false
group.cookieList.forEach((cookie)=>{if(!group.groupOptOut){cookie.cookieOptOut=false}
cookie.cookieSelected=true})})} else{this.data.cookieGroups.forEach((group)=>{if(!group.groupOptOut){group.groupSelected=true
group.groupIndeterminate=false
group.cookieList.forEach((cookie)=>{cookie.cookieOptOut=false
cookie.cookieSelected=true})} else if(group.groupSelected===true){group.groupIndeterminate=false
group.cookieList.forEach((cookie)=>{cookie.cookieSelected=true})} else{group.cookieList.forEach((cookie)=>{if(!cookie.cookieOptOut){cookie.cookieSelected=group.groupSelected}})
let allCookiesChecked=true
let allCookiesUnchecked=true
group.cookieList.forEach((cookie)=>{if(!cookie.cookieSelected){allCookiesChecked=false}
if(cookie.cookieSelected){allCookiesUnchecked=false}})
if(allCookiesChecked){group.groupSelected=true
group.groupIndeterminate=false} else if(allCookiesUnchecked){group.groupSelected=false
group.groupIndeterminate=false} else{group.groupSelected=true
group.groupIndeterminate=true}}})
let allGroupsChecked=true
let allGroupsUnchecked=true
let allGroupsDeterminate=true
this.data.cookieGroups.forEach((group)=>{if(group.groupOptOut){if(!group.groupSelected){allGroupsChecked=false}
if(group.groupSelected){allGroupsUnchecked=false}
if(group.groupIndeterminate){allGroupsDeterminate=false}}})
if(allGroupsDeterminate){if(allGroupsChecked){this.data.selectAll=true
this.data.allIndeterminate=false} else if(allGroupsUnchecked){this.data.selectAll=false
this.data.allIndeterminate=false} else{this.data.selectAll=true
this.data.allIndeterminate=true}} else{this.data.selectAll=true
this.data.allIndeterminate=true}}}
_hideCookiebar(){this.component.classList.add('d-none')
document.querySelectorAll('.redefinir-cookies').forEach((redefinir)=>{redefinir.style.display='block'})
document.querySelectorAll('.reset-cookies').forEach((reset_cookies)=>{reset_cookies.classList.remove('d-none')})}
_setRejectButton(){this.component.querySelectorAll('.actions .br-button.reject-all').forEach((button)=>{button.innerHTML=this.data.rejectButton||'Rejeitar todos'})}
_setPrimaryButton(){this.component.querySelectorAll('.actions .br-button.btn-accept').forEach((button)=>{this.data.acceptLabel?(button.innerHTML=this.data.acceptLabel):(button.innerHTML='Aceitar')
this.data.acceptLabel?button.setAttribute('aria-label',this.data.acceptLabel):buton.setAttribute('aria-label','Aceitar')
if(window.matchMedia('(max-width: 574px)').matches){button.classList.add('block')}
if(window.matchMedia('(min-width: 575px)').matches){button.classList.remove('block')}
button.addEventListener('keydown',(event)=>{if(event.key==='Tab'){if(!this.component.classList.contains('default')){this.component.focus()}}})})}
_setSecondaryButton(){this.component.querySelectorAll('.actions .br-button.btn-manage').forEach((button)=>{button.classList.add('manage')
this.data.allOptOut?this.data.optOutButton?(button.innerHTML=this.data.optOutButton):(button.innerHTML='Definir Cookies'):this.data.optInButton?(button.innerHTML=this.data.optInButton):(button.innerHTML='Ver Política de Cookies')
this.data.allOptOut?this.data.optOutButton?button.setAttribute('aria-label',this.data.optOutButton):button.setAttribute('aria-label','Definir Cookies'):this.data.optInButton?button.setAttribute('aria-label',this.data.optInButton):button.setAttribute('aria-label','Ver Política de Cookies')
if(window.matchMedia('(max-width: 574px)').matches){button.classList.add('block')}
if(window.matchMedia('(min-width: 575px)').matches){button.classList.remove('block')}
button.addEventListener('click',()=>{this.component.classList.remove('default')
button.classList.add('d-none')
this.component.focus()
document.body.style.overflowY='hidden'
this._setOpenView()
document.querySelectorAll('.br-button.btn-accept').forEach((button)=>{button.innerHTML=this.data.acceptButton
button.setAttribute('aria-label',this.data.acceptButton)})})})}
_setCloseButton(){this.component.querySelectorAll('.br-modal-header .br-button.close').forEach((closeButton)=>{closeButton.addEventListener('click',()=>{this.component.classList.add('default')
if(this.mode==='open'){this._hideCookiebar()
this._resetCookiebar()} else{fetch(config_url).then(res=>res.json()).then((out)=>{var msgs=JSON.parse(out['mensagens'])
var initial_config=msgs[0]
var groups=initial_config.cookieGroups
for(var i=0;i<groups.length;i++){var group_selected=groups[i].groupSelected
var group=this.data.cookieGroups[i]
group.groupSelected=group_selected
var cookie_list=group.cookieList
cookie_list.forEach(cookie=>{cookie.cookieSelected=group_selected})
var checkbox_id=`check-group-${i}`
var checkbox=document.getElementById(checkbox_id)
if(checkbox&&group_selected){checkbox.checked=true
this._removeGroupAlertMessage(i)
checkbox.nextElementSibling.innerHTML=this.data.unselectAllGroupLabel?this.data.unselectAllGroupLabel:'Desselecionar toda classe'
checkbox.setAttribute('aria-label',this.data.unselectAllGroupLabel?this.data.unselectAllGroupLabel:'Desselecionar toda classe')} else if(checkbox&&!group_selected){checkbox.checked=false
this._insertGroupAlertMessage(i)
checkbox.nextElementSibling.innerHTML=this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe'
checkbox.setAttribute('aria-label',this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe')}}})}
this.component.querySelector('.actions .br-button.btn-manage').classList.remove('d-none')
document.body.style.overflowY='auto'
this._setCloseView()
document.querySelectorAll('.br-button.btn-accept').forEach((button)=>{button.innerHTML=this.data.acceptLabel
button.setAttribute('aria-label',this.data.acceptLabel)})})})}
_setOpenView(){const wrapper=this.component.querySelector('.br-modal > .br-card .wrapper')
const containerFluid=this.component.querySelector('.br-modal > .br-card .container')
const modalFooter=this.component.querySelector('.br-modal > .br-card .br-modal-footer')
const padding=window.getComputedStyle(containerFluid,null).getPropertyValue('padding-top').match(/\d+/)
const height=`${window.innerHeight-padding * 2-modalFooter.offsetHeight}px`
wrapper.style.height=height}
_setCloseView(){const wrapper=this.component.querySelector('.br-modal > .br-card .wrapper')
wrapper.removeAttribute('style')}
_setResize(){window.addEventListener('resize',()=>{if(!this.component.classList.contains('default')){this._setOpenView()}
this.component.querySelectorAll('.actions .br-button').forEach((button)=>{if(window.matchMedia('(max-width: 574px)').matches){button.classList.add('block')}
if(window.matchMedia('(min-width: 575px)').matches){button.classList.remove('block')}})})}
_setInfoText(){this.component.querySelectorAll('.br-modal-body .info-text').forEach((infoText)=>{infoText.innerHTML=this.data.infoText})}
_setMainTitle(){this.component.querySelectorAll('.br-modal-header .br-modal-title').forEach((mainTitle)=>{const title=`<div class="row"><div class="col"><h1>${this.data.mainTitle}</h1></div><button class="br-button close circle small float-right" type="button" data-dismiss="br-modal" aria-label="${this.data.closeLabel?this.data.closeLabel:'fechar'}"><i class="fa fa-times"></i></button></div>`
mainTitle.innerHTML=title})}
_setListHeader(){this.component.querySelectorAll('.br-modal-body .main-content').forEach((mainContent)=>{const listHeader=`<div class="header d-none"><div class="row justify-content-between flex-fill"><div class="col-12 col-sm align-self-center mb-2"><div class="title">${this.data.cookieGroupsLabel?this.data.cookieGroupsLabel:'Classes de cookies'}</div></div><div class="col-12 align-self-center">${this.data.allOptOut?this._createCheckAll(this.data.selectAll,this.data.allIndeterminate):''}</div><div class="col-12 text-sm-right message mt-1">${(!this.data.selectAll||this.data.allIndeterminate)&&this.data.allAlertMessage?`<span class="feedback warning" role="alert">${this.data.allAlertMessage}</span>`:''}</div></div></div>`
mainContent.insertAdjacentHTML('afterbegin',listHeader)})}
_setEntryText(){this.component.querySelectorAll('.br-modal-header .entry-text').forEach((entryText)=>{entryText.innerHTML=this.data.entryText})}
_setGroupInfo(){this.component.querySelectorAll('.br-modal-body .main-content').forEach((mainContent)=>{this.data.cookieGroups.forEach((item,index)=>{const group=`<div class="br-item group-info"><div class="row mb-1"><div><span class="group-name" title="Expandir">${item.groupName}</span></div><div>${item.groupOptOut?this._createCheckGroup(index,'group-',item.groupSelected,item.groupIndeterminate,false):this._createCheckGroup(index,'group-',true,false,true)}</div></div><div class="row"><div class="col-12 message mt-1 mb-1">${(!item.groupSelected||item.groupIndeterminate)&&item.groupAlertMessage?`<span class="feedback warning" role="alert">${item.groupAlertMessage}<a href="#" class="feedback-warning-close" onclick="$(this).parent().remove();return false;">x</a></span>`:''}</div></div><div class="row"><div class="col"><p class="group-description">${item.groupText}</p></div></div></div><div class="br-list cookie-info"><div class="br-item"><div class="row"><div class="col">${this._buildCookieInfo(index,item.cookieList)}</div></div></div></div>`
mainContent.insertAdjacentHTML('beforeend',group)})})}
_setNotification(){if(this.data.noteList&&this.data.noteList.length>0){this.component.querySelectorAll('.br-modal-body .main-content').forEach((mainContent)=>{const note=`<hr><div class="br-item open"><div class="row"><div><span class="group-name" title="Expandir">${this.data.noteTitle}</span></div><div class="col-auto"><button class="br-button circle small" type="button" title="Expandir" aria-label="Expandir"><i class="fa fa-angle-down" aria-hidden="true"></i></button></div></div></div><div class="br-list">${this._buildNotificationInfo()}</div><hr>`
mainContent.insertAdjacentHTML('beforeend',note)})}}
_setLinks(){if(this.data.links&&this.data.links.length>0){this.component.querySelectorAll('.br-modal-body').forEach((modalHeader)=>{const link=`<div class="br-list parallel-content">${this._buildLinkInfo()}</div>`
modalHeader.insertAdjacentHTML('beforeend',link)})}}
_buildCookieInfo(groupIndex,cookieList){let cookieInfo=''
cookieList.forEach((cookie,index)=>{cookieInfo+=`<div class="br-card"><div class="card-content"><div class="row mb-1"><div class="col-12 text-right">${cookie.cookieOptOut?this._createSwitchButton(index,`cookie-${groupIndex}`,cookie.cookieSelected):''}</div><div class="col-12 message text-right mb-1 mt-1">${!cookie.cookieSelected&&cookie.alertMessage?`<span class="feedback warning" role="alert">${this.data.cookieGroups[groupIndex].cookieList[index].alertMessage}<a href="#" class="feedback-warning-close" onclick="$(this).parent().remove();return false;">x</a></span>`:''}</div></div><div class="row"><div class="fixed-width cookie-term"><span>${this.data.cookieNameLabel?this.data.cookieNameLabel:'Cookies'}</span></div><div class="col-12 col-sm mb-1 cookie-data"><span>${cookie.cookieName}</span></div></div><div class="row"><div class="fixed-width cookie-term"><span>${this.data.expiresLabel?this.data.expiresLabel:'Vencimento'}</span></div><div class="col-12 col-sm mb-1 cookie-data"><span>${cookie.expires}</span></div></div><div class="row"><div class="fixed-width cookie-term"><span>${this.data.domainLabel?this.data.domainLabel:'Domínio'}</span></div><div class="col-12 col-sm mb-1 cookie-data"><span>${cookie.domain}</span></div></div><div class="row"><div class="fixed-width cookie-term"><span>${this.data.enterpriseLabel?this.data.enterpriseLabel:'Empresa'}</span></div><div class="col-12 col-sm mb-1 cookie-data"><span>${cookie.entreprise}</span></div></div><div class="row"><div class="fixed-width cookie-term"><span>${this.data.purposeLabel?this.data.purposeLabel:'Finalidade'}</span></div><div class="col-12 col-sm mb-1 cookie-data"><span>${cookie.purpose}</span></div></div><div class="row"><div class="fixed-width cookie-term"><span>${this.data.descriptionLabel?this.data.descriptionLabel:'Descrição'}</span></div><div class="col-12 col-sm mb-1 cookie-data"><span>${cookie.description}</span></div></div></div></div>`})
return cookieInfo}
_createCheckAll(checked,indeterminate){const checkAll=`<div class="br-checkbox-govbr"><input id="check-all" name="check-all" type="checkbox" aria-label="${this._setCheckAllLabel(checked,indeterminate)}" ${checked?'checked':''} ${indeterminate?'indeterminate':''} tabindex="0"/><label for="check-all">${this._setCheckAllLabel(checked,indeterminate)}</label></div>`
return checkAll}
_setCheckAllLabel(checked,indeterminate){if(indeterminate||!checked){return this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo'} else{return this.data.unselectAllLabel?this.data.unselectAllLabel:'Desselecionar tudo'}}
_createCheckGroup(index,piece,checked,indeterminate,disabled){const checkGroup=`<div class="br-checkbox-govbr"><input id="check-${piece + index}" name="check-${piece+index}" type="checkbox" aria-label="${this._setCheckGroupLabel(checked,indeterminate)}" ${checked?'checked':''} ${indeterminate?'indeterminate':''} ${disabled?'disabled':''} tabindex="0"/><label for="check-${piece+index}">${this._setCheckGroupLabel(checked,indeterminate)}</label></div>`
return checkGroup}
_setCheckGroupLabel(checked,indeterminate){if(indeterminate||!checked){return this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe'} else{return this.data.unselectAllGroupLabel?this.data.unselectAllGroupLabel:'Desselecionar toda classe'}}
_createSwitchButton(index,piece,selected){const switchButton=`<div class="br-switch small icon"><input id="switch-${piece+index}" type="checkbox" ${selected?'checked':''} tabindex="-1"><label for="switch-${piece + index}" aria-label="${selected?this.data.onLabel?this.data.onLabel:'Ligado':this.data.offLabel?this.data.offLabel:'Desligado'}"></label><div class="switch-data" data-enabled="${this.data.onLabel?this.data.onLabel:'Ligado'}" data-disabled="${this.data.offLabel?this.data.offLabel:'Desligado'}"></div></div>`
return switchButton}
_buildNotificationInfo(){let noteInfo=''
this.data.noteList.forEach((item)=>{noteInfo+=`<div class="br-item notes"><div class="row"><div class="col"><p>${item.question}</p><p>${item.answer}</p></div></div></div>`})
return noteInfo}
_buildLinkInfo(){let linkInfo=''
this.data.links.forEach((link)=>{linkInfo+=`<div class="br-item"><div class="row"><div class="col"><a href="${link.url}">${link.name}<i class="fa fa-external-link" aria-hidden="true"></i></a></div></div></div>`})
return linkInfo}
_insertAllSelectedAlertMessage(){this._removeAllSelectedAlertMessage()
if(this.data.allAlertMessage&&this.data.allAlertMessage!==''){const messageTemplate=`<span class="feedback warning" role="alert">${this.data.allAlertMessage}<a href="#" class="feedback-warning-close" onclick="$(this).parent().remove();return false;">x</a></span>`
const place=this.component.querySelector('.header .row:nth-child(1) div:nth-child(3)')
place.insertAdjacentHTML('afterbegin',messageTemplate)}}
_insertGroupAlertMessage(indexGroup){this._removeGroupAlertMessage(indexGroup)
if(this.data.cookieGroups[indexGroup].groupAlertMessage&&this.data.cookieGroups[indexGroup].groupAlertMessage!==''){const messageTemplate=`<span class="feedback warning" role="alert">${this.data.cookieGroups[indexGroup].groupAlertMessage}<a href="#" class="feedback-warning-close" onclick="$(this).parent().remove();return false;">x</a></span>`
for(const [indexGr,group] of this.component.querySelectorAll('.main-content .group-info').entries()){if(indexGr===indexGroup){const place=group.querySelector('.row:nth-child(2) div:nth-child(1)')
place.insertAdjacentHTML('afterbegin',messageTemplate)
break}}}}
_insertAlertMessage(indexGroup,indexCookie){if(this.data.cookieGroups[indexGroup].cookieList[indexCookie].alertMessage&&this.data.cookieGroups[indexGroup].cookieList[indexCookie].alertMessage!==''){const messageTemplate=`<span class="feedback warning" role="alert">${this.data.cookieGroups[indexGroup].cookieList[indexCookie].alertMessage}<a href="#" class="feedback-warning-close" onclick="$(this).parent().remove();return false;">x</a></span>`
for(const [indexGr,group] of this.component.querySelectorAll('.main-content .group-info').entries()){if(indexGr===indexGroup){for(const [indexCo,cookie,] of group.nextElementSibling.querySelectorAll('.main-content .cookie-info .br-card').entries()){if(indexCo===indexCookie){const place=cookie.querySelector('.row:nth-child(1) div:nth-child(2)')
place.insertAdjacentHTML('afterbegin',messageTemplate)
break}}
break}}}}
_removeAllSelectedAlertMessage(){const feedback=this.component.querySelector('.header .row:nth-child(1) div:nth-child(3) .feedback')
if(feedback){feedback.remove()}}
_removeGroupAlertMessage(indexGroup){for(const [indexGr,group] of this.component.querySelectorAll('.main-content .group-info').entries()){if(indexGr===indexGroup){const feedback=group.querySelector('.row:nth-child(2) div:nth-child(1) .feedback')
if(feedback){feedback.remove()}
break}}}
_removeAlertMessage(indexGroup,indexCookie){for(const [indexGr,group] of this.component.querySelectorAll('.main-content .group-info').entries()){if(indexGr===indexGroup){for(const [indexCo,cookie,] of group.nextElementSibling.querySelectorAll('.main-content .cookie-info .br-card').entries()){if(indexCo===indexCookie){const feedback=cookie.querySelector('.row:nth-child(1) div:nth-child(2) .feedback')
if(feedback){feedback.remove()}
break}}
break}}}
_setCheckAll(){this.component.querySelectorAll('.main-content .header .br-checkbox-govbr input[type="checkbox"]').forEach((checkAll)=>{checkAll.addEventListener('click',()=>{if(checkAll.hasAttribute('indeterminate')){checkAll.checked=true
this.data.allIndeterminate=false
checkAll.removeAttribute('indeterminate')}
this.data.selectAll=checkAll.checked
if(checkAll.checked){this._removeAllSelectedAlertMessage()
checkAll.nextElementSibling.innerHTML=this.data.unselectAllLabel?this.data.unselectAllLabel:'Desselecionar tudo'
checkAll.setAttribute('aria-label',this.data.unselectAllLabel?this.data.unselectAllLabel:'Desselecionar tudo')} else{this._insertAllSelectedAlertMessage()
checkAll.nextElementSibling.innerHTML=this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo'
checkAll.setAttribute('aria-label',this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo')}
const checkAllState=checkAll.checked
this.component.querySelectorAll('.main-content .group-info .br-checkbox-govbr input[type="checkbox"]').forEach((checkGroup)=>{if(checkGroup.hasAttribute('indeterminate')){checkGroup.click()} else{if(checkAllState&&!checkGroup.checked){checkGroup.click()}
if(!checkAllState&&checkGroup.checked){checkGroup.click()}}})})})}
_setCheckGroup(){this.component.querySelectorAll('.main-content .group-info').forEach((group,groupIndex)=>{group.querySelectorAll('.br-checkbox-govbr input[type="checkbox"]').forEach((checkGroup)=>{checkGroup.addEventListener('click',()=>{if(checkGroup.hasAttribute('indeterminate')){checkGroup.checked=true
this.data.cookieGroups[groupIndex].groupIndeterminate=false
checkGroup.removeAttribute('indeterminate')}
this.data.cookieGroups[groupIndex].groupSelected=checkGroup.checked
if(checkGroup.checked){this._removeGroupAlertMessage(groupIndex)
checkGroup.nextElementSibling.innerHTML=this.data.unselectAllGroupLabel?this.data.unselectAllGroupLabel:'Desselecionar toda classe'
checkGroup.setAttribute('aria-label',this.data.unselectAllGroupLabel?this.data.unselectAllGroupLabel:'Desselecionar toda classe')} else{this._insertGroupAlertMessage(groupIndex)
checkGroup.nextElementSibling.innerHTML=this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe'
checkGroup.setAttribute('aria-label',this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe')}
const checkGroupState=checkGroup.checked
group.nextElementSibling.querySelectorAll('.cookie-info .br-card').forEach((cookie,cookieIndex)=>{if(!this.data.cookieGroups[groupIndex].cookieList[cookieIndex].cookieOptOut){this.data.cookieGroups[groupIndex].cookieList[cookieIndex].cookieSelected=checkGroup.checked
if(checkGroup.checked){this._removeAlertMessage(groupIndex,cookieIndex)} else{this._insertAlertMessage(groupIndex,cookieIndex)}} else{cookie.querySelectorAll('.br-switch input[type="checkbox"]').forEach((cookieSwitch)=>{if(checkGroupState&&!cookieSwitch.checked){cookieSwitch.click()}
if(!checkGroupState&&cookieSwitch.checked){cookieSwitch.click()}})}})
let allGroupsChecked=true
let allGroupsUnchecked=true
let allGroupsDeterminate=true
this.component.querySelectorAll('.main-content .group-info .br-checkbox-govbr input[type="checkbox"]').forEach((checkGroup)=>{if(!checkGroup.checked){allGroupsChecked=false}
if(checkGroup.checked){allGroupsUnchecked=false}
if(checkGroup.hasAttribute('indeterminate')){allGroupsDeterminate=false}})
this.component.querySelectorAll('.main-content .header .br-checkbox-govbr input[type="checkbox"]').forEach((checkAll)=>{if(allGroupsDeterminate){if(allGroupsChecked){checkAll.checked=true
this.data.selectAll=true
checkAll.removeAttribute('indeterminate')
this.data.allIndeterminate=false} else if(allGroupsUnchecked){checkAll.checked=false
this.data.selectAll=false
checkAll.removeAttribute('indeterminate')
this.data.allIndeterminate=false} else{checkAll.checked=true
this.data.selectAll=true
checkAll.setAttribute('indeterminate','')
this.data.allIndeterminate=true}} else{checkAll.checked=true
this.data.selectAll=true
checkAll.setAttribute('indeterminate','')
this.data.allIndeterminate=true}
if(!checkAll.hasAttribute('indeterminate')){if(checkAll.checked){this._removeAllSelectedAlertMessage()
checkAll.nextElementSibling.innerHTML=this.data.unselectAllLabel?this.data.unselectAllLabel:'Desselecionar tudo'
checkAll.setAttribute('aria-label',this.data.unselectAllLabel?this.data.unselectAllLabel:'Desselecionar tudo')} else{this._insertAllSelectedAlertMessage()
checkAll.nextElementSibling.innerHTML=this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo'
checkAll.setAttribute('aria-label',this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo')}} else{this._insertAllSelectedAlertMessage()
checkAll.nextElementSibling.innerHTML=this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo'
checkAll.setAttribute('aria-label',this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo')}})})})})}
_setCheckCookie(){this.component.querySelectorAll('.main-content .group-info').forEach((group,groupIndex)=>{group.nextElementSibling.querySelectorAll('.cookie-info .br-card').forEach((cookie,cookieIndex)=>{cookie.querySelectorAll('.br-switch input[type="checkbox"]').forEach((checkCookie)=>{checkCookie.addEventListener('click',()=>{this.data.cookieGroups[groupIndex].cookieList[cookieIndex].cookieSelected=checkCookie.checked
if(checkCookie.checked){checkCookie.nextElementSibling.setAttribute('aria-label',this.data.onLabel?this.data.onLabel:'Ligado')
this._removeAlertMessage(groupIndex,cookieIndex)} else{checkCookie.nextElementSibling.setAttribute('aria-label',this.data.offLabel?this.data.offLabel:'Desligado')
this._insertAlertMessage(groupIndex,cookieIndex)}
let allCookiesChecked=true
let allCookiesUnchecked=true
group.nextElementSibling.querySelectorAll('.cookie-info .br-card .br-switch input[type="checkbox"]').forEach((checkCookie)=>{if(!checkCookie.checked){allCookiesChecked=false}
if(checkCookie.checked){allCookiesUnchecked=false}})
this.data.cookieGroups[groupIndex].cookieList.forEach((cookie)=>{if(!cookie.cookieOptOut){if(!cookie.cookieSelected){allCookiesChecked=false}
if(cookie.cookieSelected){allCookiesUnchecked=false}}})
group.querySelectorAll('.br-checkbox-govbr input[type="checkbox"]').forEach((checkGroup)=>{if(allCookiesChecked){checkGroup.checked=true
this.data.cookieGroups[groupIndex].groupSelected=true
checkGroup.removeAttribute('indeterminate')
this.data.cookieGroups[groupIndex].groupIndeterminate=false} else if(allCookiesUnchecked){checkGroup.checked=false
this.data.cookieGroups[groupIndex].groupSelected=false
checkGroup.removeAttribute('indeterminate')
this.data.cookieGroups[groupIndex].groupIndeterminate=false} else{checkGroup.checked=true
this.data.cookieGroups[groupIndex].groupSelected=true
checkGroup.setAttribute('indeterminate','')
this.data.cookieGroups[groupIndex].groupIndeterminate=true}
if(!checkGroup.hasAttribute('indeterminate')){if(checkGroup.checked){this._removeGroupAlertMessage(groupIndex)
checkGroup.nextElementSibling.innerHTML=this.data.unselectAllGroupLabel?this.data.unselectAllGroupLabel:'Desselecionar toda classe'
checkGroup.setAttribute('aria-label',this.data.unselectAllGroupLabel?this.data.unselectAllGroupLabel:'Desselecionar toda classe')} else{this._insertGroupAlertMessage(groupIndex)
checkGroup.nextElementSibling.innerHTML=this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe'
checkGroup.setAttribute('aria-label',this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe')}} else{this._insertGroupAlertMessage(groupIndex)
checkGroup.nextElementSibling.innerHTML=this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe'
checkGroup.setAttribute('aria-label',this.data.selectAllGroupLabel?this.data.selectAllGroupLabel:'Selecionar toda classe')}})
let allGroupsChecked=true
let allGroupsUnchecked=true
let allGroupsDeterminate=true
this.component.querySelectorAll('.main-content .group-info .br-checkbox-govbr input[type="checkbox"]').forEach((checkGroup)=>{if(!checkGroup.checked){allGroupsChecked=false}
if(checkGroup.checked){allGroupsUnchecked=false}
if(checkGroup.hasAttribute('indeterminate')){allGroupsDeterminate=false}})
this.component.querySelectorAll('.main-content .header .br-checkbox-govbr input[type="checkbox"]').forEach((checkAll)=>{if(allGroupsDeterminate){if(allGroupsChecked){checkAll.checked=true
this.data.selectAll=true
checkAll.removeAttribute('indeterminate')
this.data.allIndeterminate=false} else if(allGroupsUnchecked){checkAll.checked=false
this.data.selectAll=false
checkAll.removeAttribute('indeterminate')
this.data.allIndeterminate=false} else{checkAll.checked=true
this.data.selectAll=true
checkAll.setAttribute('indeterminate','')
this.data.allIndeterminate=true}} else{checkAll.checked=true
this.data.selectAll=true
checkAll.setAttribute('indeterminate','')
this.data.allIndeterminate=true}
if(!checkAll.hasAttribute('indeterminate')){if(checkAll.checked){this._removeAllSelectedAlertMessage()
checkAll.nextElementSibling.innerHTML=this.data.unselectAllLabel?this.data.unselectAllLabel:'Desselecionar tudo'
checkAll.setAttribute('aria-label',this.data.unselectAllLabel?this.data.unselectAllLabel:'Desselecionar tudo')} else{this._insertAllSelectedAlertMessage()
checkAll.nextElementSibling.innerHTML=this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo'
checkAll.setAttribute('aria-label',this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo')}} else{this._insertAllSelectedAlertMessage()
checkAll.nextElementSibling.innerHTML=this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo'
checkAll.setAttribute('aria-label',this.data.selectAllLabel?this.data.selectAllLabel:'Selecionar tudo')}})})})})})}
_setCookiesChecked(groupData){let count=0
groupData.cookieList.forEach((cookie)=>{if(cookie.cookieSelected){count=count+1}})
return `${count} de `}
_updateCookiesChecked(groupData,groupIndexData){let count=0
groupData.cookieList.forEach((cookie)=>{if(cookie.cookieSelected){count=count+1}})
this.component.querySelectorAll('.main-content .group-info').forEach((group,groupIndex)=>{if(groupIndex===groupIndexData){group.querySelector('.cookies-checked').innerHTML=`(${count} de `}})}
_setGroupBehavior(){this.component.querySelectorAll('.main-content .br-item .br-button, .main-content .br-item .group-name, .main-content .br-item .group-size ').forEach((clickArea)=>{clickArea.addEventListener('click',()=>{let element=clickArea
while(!element.classList.contains('br-item')){element=element.parentNode}
const icon=element.querySelector('.br-button i.fa')
if(element.classList.contains('open')){element.classList.remove('open')
icon.classList.remove('fa-angle-up')
icon.classList.add('fa-angle-down')
element.querySelectorAll('.br-button, .group-name, .group-size').forEach((item)=>{item.setAttribute('title','Expandir')
item.setAttribute('aria-label','Expandir')})} else{element.classList.add('open')
icon.classList.remove('fa-angle-down')
icon.classList.add('fa-angle-up')
element.querySelectorAll('.br-button, .group-name, .group-size').forEach((item)=>{item.setAttribute('title','Retrair')
item.setAttribute('aria-label','Retrair')})
setTimeout(()=>{this.component.querySelectorAll('.wrapper').forEach(()=>{setTimeout(()=>{element.scrollIntoView({behavior:'smooth',block:'start',})},150)},5000)})}
if(element.classList.contains('open')){element.nextElementSibling.querySelectorAll('.br-card .br-switch input[type="checkbox"]').forEach((checkSwitch)=>{checkSwitch.setAttribute('tabindex','0')})} else{element.nextElementSibling.querySelectorAll('.br-card .br-switch input[type="checkbox"]').forEach((checkSwitch)=>{checkSwitch.setAttribute('tabindex','-1')})}})})}
_saveJSON(){this._hideCookiebar()
this._resetCookiebar()
document.body.style.overflowY='auto'
this.callback(this._setOutputJSON())
window.location=window.location.href}
_setAcceptBehavior(){const acceptButton=this.component.querySelector('.actions .br-button.btn-accept')
acceptButton.addEventListener('click',()=>{if(this.component.classList.contains('default')){this.component.querySelectorAll('.group-info input[type="checkbox"]:not([disabled])').forEach((checkbox)=>{if(checkbox.hasAttribute('indeterminate')){checkbox.click()}
if(!checkbox.checked){checkbox.click()}})}
this._saveJSON()})}
_setRejectBehavior(){const rejectButton=this.component.querySelector('.br-button.reject-all')
rejectButton.addEventListener('click',()=>{this.component.querySelectorAll('.group-info input[type="checkbox"]:not([disabled])').forEach((checkbox)=>{if(checkbox.hasAttribute('indeterminate')){checkbox.click()}
if(checkbox.checked){checkbox.click()}})
this._saveJSON()})}
_setOutputJSON(){this.output={}
this.output.v=version
this.output.g=[]
this.data.cookieGroups.forEach((group)=>{const cookies=[]
group.cookieList.forEach((cookie)=>{cookies.push(cookie.cookieId)})
localStorage.setItem(group.groupId,JSON.stringify(cookies));this.output.g.push({id:group.groupId,on:group.groupIndeterminate?'indeterminate':group.groupSelected?true:false,})})
return JSON.stringify(this.output)}
_resetCookiebar(){const modalHeader=`<div class="br-modal-header entry-content"><div class="br-modal-title"></div><div class="last-update"></div><div class="entry-text"></div></div>`
const modalBody=`<div class="br-modal-body"><div class="info-text"></div><div class="br-list main-content"></div></div>`
const modalFooter=`<div class="br-modal-footer actions"><button class="br-button small btn-manage" type="button" aria-label="Definir Cookies"></button><button class="br-button secondary small reject-all" type="button" aria-label="Rejeitar">Rejeitar</button><button class="br-button secondary small btn-accept" type="button" aria-label="Aceitar"></button></div>`
document.querySelectorAll('.br-cookiebar .br-modal .br-card > div:first-child .br-modal-footer').forEach((item)=>{item.parentElement.insertAdjacentHTML('beforeend',modalFooter)
item.remove()})
document.querySelectorAll('.br-cookiebar .br-modal .wrapper .br-modal-body').forEach((item)=>{item.parentElement.insertAdjacentHTML('afterbegin',modalBody)
item.remove()})
document.querySelectorAll('.br-cookiebar .br-modal .wrapper .br-modal-header').forEach((item)=>{item.parentElement.insertAdjacentHTML('afterbegin',modalHeader)
item.remove()})}}
function _updateOutputJSON(json){var cookie_v2=('; '+document.cookie).split(`lgpd-cookie-v2=`).pop().split(';')[0];if(cookie_v2){var json_cookie_v2=JSON.parse(cookie_v2);if(json_cookie_v2){var groups_cookie=json_cookie_v2['g'];var groups_default=json['cookieGroups'];for(var i=0,li=groups_cookie.length;i<li;i++){for(var j=0,lj=groups_default.length;j<lj;j++){if(groups_cookie[i]['id']==groups_default[j]['groupId']){var group_selected=false
if(groups_cookie[i]['on']===1||groups_cookie[i]['on']===0){group_selected=groups_cookie[i]['on']==1?true:false}
if(groups_cookie[i]['on']===true||groups_cookie[i]['on']===false){group_selected=groups_cookie[i]['on']==true?true:false}
json['cookieGroups'][j]['groupSelected']=group_selected
var cookies=groups_default[i]['cookieList']
for(var k=0,lk=cookies.length;k<lk;k++){json['cookieGroups'][i]['cookieList'][k]['cookieSelected']=group_selected}}}}}}
var output={}
output.v=version
output.g=[]
json.cookieGroups.forEach((group)=>{const cookies=[]
group.cookieList.forEach((cookie)=>{cookies.push(cookie.cookieId)})
localStorage.setItem(group.groupId,JSON.stringify(cookies));output.g.push({id:group.groupId,on:group.groupIndeterminate?'indeterminate':group.groupSelected?true:false,})})
return JSON.stringify(output)}
$(document).ready(function(){document.cookie='lgpd_cookie_status= ; expires=Thu, 01 Jan 2000 00:00:01 GMT; path=/';var habilitar=false;var idioma='';var mensagens='';var cookiename='lgpd-cookie-v2';var config_url=urlConfiguracao();fetch(config_url).then(res=>res.json()).then((out)=>{habilitar=out['habilitar'];idioma=out['idioma'];mensagens=out['mensagens'];cookiename='lgpd-cookie-v2';if(habilitar){var cookieinformation=getCookie(cookiename);if(typeof cookieinformation!=='undefined'){;var versao=JSON.parse(cookieinformation)['v'];if(versao!=version){var json={}
var json_mensagens=JSON.parse(mensagens)
if(json_mensagens){var mensagens_idioma=json_mensagens.filter((obj)=>{return obj.lang===idioma})
if(mensagens_idioma.length>0){json=mensagens_idioma[0]} else{json=json_mensagens[0]}}
if(json!={}&&json!='undefined'){gravarCookie(_updateOutputJSON(json));window.location=window.location.href}}
_createEventIcone()}
else{novaCookieBar(idioma,mensagens,'default')}}}).catch(err=>console.error(err));if(!getCookie('lgpd-cookie-v2')){$('.reset-cookies').addClass('d-none')}});
