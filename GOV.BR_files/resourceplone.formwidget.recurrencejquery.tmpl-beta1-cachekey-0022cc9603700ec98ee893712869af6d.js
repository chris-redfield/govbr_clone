
/* Merged Plone Javascript file
 * This file is dynamically assembled from separate parts.
 * Some of these parts have 3rd party licenses or copyright information attached
 * Such information is valid for that section,
 * not for the entire composite file
 * originating files are separated by - filename.js -
 */

/* - ++resource++plone.formwidget.recurrence/jquery.tmpl-beta1.js - */
// https://www.gov.br/portal_javascripts/++resource++plone.formwidget.recurrence/jquery.tmpl-beta1.js?original=1
(function(jQuery,undefined){var oldManip=jQuery.fn.domManip,tmplItmAtt="_tmplitem",htmlExpr=/^[^<]*(<[\w\W]+>)[^>]*$|\{\{\! /,newTmplItems={},wrappedItems={},appendToTmplItems,topTmplItem={key:0,data:{}},itemKey=0,cloneIndex=0,stack=[];
function newTmplItem(options,parentItem,fn,data){var newItem={data:data||(parentItem?parentItem.data:{}),_wrap:parentItem?parentItem._wrap:null,tmpl:null,parent:parentItem||null,nodes:[],calls:tiCalls,nest:tiNest,wrap:tiWrap,html:tiHtml,update:tiUpdate};if(options){jQuery.extend(newItem,options,{nodes:[],parent:parentItem})}
if(fn){newItem.tmpl=fn;newItem._ctnt=newItem._ctnt||newItem.tmpl(jQuery,newItem);newItem.key=++itemKey;(stack.length?wrappedItems:newTmplItems)[itemKey]=newItem}
return newItem}
jQuery.each({appendTo:"append",prependTo:"prepend",insertBefore:"before",insertAfter:"after",replaceAll:"replaceWith"}, function(name,original){jQuery.fn[name]=function(selector){var ret=[],insert=jQuery(selector),elems,i,l,tmplItems,parent=this.length===1&&this[0].parentNode;appendToTmplItems=newTmplItems||{};if(parent&&parent.nodeType===11&&parent.childNodes.length===1&&insert.length===1){insert[original](this[0]);ret=this} else{for(i=0,l=insert.length;i<l;i++){cloneIndex=i;elems=(i>0?this.clone(true):this).get();jQuery.fn[original].apply(jQuery(insert[i]),elems);ret=ret.concat(elems)}
cloneIndex=0;ret=this.pushStack(ret,name,insert.selector)}
tmplItems=appendToTmplItems;appendToTmplItems=null;jQuery.tmpl.complete(tmplItems);return ret}});jQuery.fn.extend({tmpl: function(data,options,parentItem){return jQuery.tmpl(this[0],data,options,parentItem)},tmplItem: function(){return jQuery.tmplItem(this[0])},template: function(name){return jQuery.template(name,this[0])},domManip: function(args,table,callback,options){if(args[0]&&args[0].nodeType){var dmArgs=jQuery.makeArray(arguments),argsLength=args.length,i=0,tmplItem;while(i<argsLength&&!(tmplItem=jQuery.data(args[i++],"tmplItem"))){}
if(argsLength>1){dmArgs[0]=[jQuery.makeArray(args)]}
if(tmplItem&&cloneIndex){dmArgs[2]=function(fragClone){jQuery.tmpl.afterManip(this,fragClone,callback)}}
oldManip.apply(this,dmArgs)} else{oldManip.apply(this,arguments)}
cloneIndex=0;if(!appendToTmplItems){jQuery.tmpl.complete(newTmplItems)}
return this}});jQuery.extend({tmpl: function(tmpl,data,options,parentItem){var ret,topLevel=!parentItem;if(topLevel){parentItem=topTmplItem;tmpl=jQuery.template[tmpl]||jQuery.template(null,tmpl);wrappedItems={}} else if(!tmpl){tmpl=parentItem.tmpl;newTmplItems[parentItem.key]=parentItem;parentItem.nodes=[];if(parentItem.wrapped){updateWrapped(parentItem,parentItem.wrapped)}
return jQuery(build(parentItem,null,parentItem.tmpl(jQuery,parentItem)))}
if(!tmpl){return []}
if(typeof data==="function"){data=data.call(parentItem||{})}
if(options&&options.wrapped){updateWrapped(options,options.wrapped)}
ret=jQuery.isArray(data)?jQuery.map(data, function(dataItem){return dataItem?newTmplItem(options,parentItem,tmpl,dataItem):null}):[newTmplItem(options,parentItem,tmpl,data)];return topLevel?jQuery(build(parentItem,null,ret)):ret},tmplItem: function(elem){var tmplItem;if(elem instanceof jQuery){elem=elem[0]}
while(elem&&elem.nodeType===1&&!(tmplItem=jQuery.data(elem,"tmplItem"))&&(elem=elem.parentNode)){}
return tmplItem||topTmplItem},template: function(name,tmpl){if(tmpl){if(typeof tmpl==="string"){tmpl=buildTmplFn(tmpl)} else if(tmpl instanceof jQuery){tmpl=tmpl[0]||{}}
if(tmpl.nodeType){tmpl=jQuery.data(tmpl,"tmpl")||jQuery.data(tmpl,"tmpl",buildTmplFn(tmpl.innerHTML))}
return typeof name==="string"?(jQuery.template[name]=tmpl):tmpl}
return name?(typeof name!=="string"?jQuery.template(null,name):(jQuery.template[name]||jQuery.template(null,htmlExpr.test(name)?name:jQuery(name)))):null},encode: function(text){return(""+text).split("<").join("&lt;").split(">").join("&gt;").split('"').join("&#34;").split("'").join("&#39;")}});jQuery.extend(jQuery.tmpl,{tag:{"tmpl":{_default:{$2:"null"},open:"if($notnull_1){_=_.concat($item.nest($1,$2));}"},"wrap":{_default:{$2:"null"},open:"$item.calls(_,$1,$2);_=[];",close:"call=$item.calls();_=call._.concat($item.wrap(call,_));"},"each":{_default:{$2:"$index, $value"},open:"if($notnull_1){$.each($1a,function($2){with(this){",close:"}});}"},"if":{open:"if(($notnull_1) && $1a){",close:"}"},"else":{_default:{$1:"true"},open:"}else if(($notnull_1) && $1a){"},"html":{open:"if($notnull_1){_.push($1a);}"},"=":{_default:{$1:"$data"},open:"if($notnull_1){_.push($.encode($1a));}"},"!":{open:""}},complete: function(items){newTmplItems={}},afterManip: function afterManip(elem,fragClone,callback){var content=fragClone.nodeType===11?jQuery.makeArray(fragClone.childNodes):fragClone.nodeType===1?[fragClone]:[];callback.call(elem,fragClone);storeTmplItems(content);cloneIndex++}});
function build(tmplItem,nested,content){var frag,ret=content?jQuery.map(content, function(item){return(typeof item==="string")?(tmplItem.key?item.replace(/(<\w+)(?=[\s>])(?![^>]*_tmplitem)([^>]*)/g,"$1 "+tmplItmAtt+"=\""+tmplItem.key+"\" $2"):item):build(item,tmplItem,item._ctnt)}):tmplItem;if(nested){return ret}
ret=ret.join("");ret.replace(/^\s*([^<\s][^<]*)?(<[\w\W]+>)([^>]*[^>\s])?\s*$/, function(all,before,middle,after){frag=jQuery(middle).get();storeTmplItems(frag);if(before){frag=unencode(before).concat(frag)}
if(after){frag=frag.concat(unencode(after))}});return frag?frag:unencode(ret)}
function unencode(text){var el=document.createElement("div");el.innerHTML=text;return jQuery.makeArray(el.childNodes)}
function buildTmplFn(markup){return new Function("jQuery","$item","var $=jQuery,call,_=[],$data=$item.data;"+"with($data){_.push('"+jQuery.trim(markup).replace(/([\\'])/g,"\\$1").replace(/[\r\t\n]/g," ").replace(/\$\{([^\}]*)\}/g,"{{= $1}}").replace(/\{\{(\/?)(\w+|.)(?:\(((?:[^\}]|\}(?!\}))*?)?\))?(?:\s+(.*?)?)?(\(((?:[^\}]|\}(?!\}))*?)\))?\s*\}\}/g,
function(all,slash,type,fnargs,target,parens,args){var tag=jQuery.tmpl.tag[type],def,expr,exprAutoFnDetect;if(!tag){throw "Template command not found: "+type}
def=tag._default||[];if(parens&&!/\w$/.test(target)){target+=parens;parens=""}
if(target){target=unescape(target);args=args?(","+unescape(args)+")"):(parens?")":"");expr=parens?(target.indexOf(".")>-1?target+parens:("("+target+").call($item"+args)):target;exprAutoFnDetect=parens?expr:"(typeof("+target+")==='function'?("+target+").call($item):("+target+"))"} else{exprAutoFnDetect=expr=def.$1||"null"}
fnargs=unescape(fnargs);return "');"+tag[slash?"close":"open"].split("$notnull_1").join(target?"typeof("+target+")!=='undefined' && ("+target+")!=null":"true").split("$1a").join(exprAutoFnDetect).split("$1").join(expr).split("$2").join(fnargs?fnargs.replace(/\s*([^\(]+)\s*(\((.*?)\))?/g, function(all,name,parens,params){params=params?(","+params+")"):(parens?")":"");return params?("("+name+").call($item"+params):all}):(def.$2||""))+"_.push('"})+"');}return _;")}
function updateWrapped(options,wrapped){options._wrap=build(options,true,jQuery.isArray(wrapped)?wrapped:[htmlExpr.test(wrapped)?wrapped:jQuery(wrapped).html()]).join("")}
function unescape(args){return args?args.replace(/\\'/g, "'").replace(/\\\\/g, "\\"):null}
function outerHtml(elem){var div=document.createElement("div");div.appendChild(elem.cloneNode(true));return div.innerHTML}
function storeTmplItems(content){var keySuffix="_"+cloneIndex,elem,elems,newClonedItems={},i,l,m;for(i=0,l=content.length;i<l;i++){if((elem=content[i]).nodeType!==1){continue}
elems=elem.getElementsByTagName("*");for(m=elems.length-1;m>=0;m--){processItemKey(elems[m])}
processItemKey(elem)}
function processItemKey(el){var pntKey,pntNode=el,pntItem,tmplItem,key;if((key=el.getAttribute(tmplItmAtt))){while(pntNode.parentNode&&(pntNode=pntNode.parentNode).nodeType===1&&!(pntKey=pntNode.getAttribute(tmplItmAtt))){}
if(pntKey!==key){pntNode=pntNode.parentNode?(pntNode.nodeType===11?0:(pntNode.getAttribute(tmplItmAtt)||0)):0;if(!(tmplItem=newTmplItems[key])){tmplItem=wrappedItems[key];tmplItem=newTmplItem(tmplItem,newTmplItems[pntNode]||wrappedItems[pntNode],null,true);tmplItem.key=++itemKey;newTmplItems[itemKey]=tmplItem}
if(cloneIndex){cloneTmplItem(key)}}
el.removeAttribute(tmplItmAtt)} else if(cloneIndex&&(tmplItem=jQuery.data(el,"tmplItem"))){cloneTmplItem(tmplItem.key);newTmplItems[tmplItem.key]=tmplItem;pntNode=jQuery.data(el.parentNode,"tmplItem");pntNode=pntNode?pntNode.key:0}
if(tmplItem){pntItem=tmplItem;while(pntItem&&pntItem.key!=pntNode){pntItem.nodes.push(el);pntItem=pntItem.parent}
delete tmplItem._ctnt;delete tmplItem._wrap;jQuery.data(el,"tmplItem",tmplItem)}
function cloneTmplItem(key){key=key+keySuffix;tmplItem=newClonedItems[key]=(newClonedItems[key]||newTmplItem(tmplItem,newTmplItems[tmplItem.parent.key+keySuffix]||tmplItem.parent,null,true))}}}
function tiCalls(content,tmpl,data,options){if(!content){return stack.pop()}
stack.push({_:content,tmpl:tmpl,item:this,data:data,options:options})}
function tiNest(tmpl,data,options){return jQuery.tmpl(jQuery.template(tmpl),data,options,this)}
function tiWrap(call,wrapped){var options=call.options||{};options.wrapped=wrapped;return jQuery.tmpl(jQuery.template(call.tmpl),call.data,options,call.item)}
function tiHtml(filter,textOnly){var wrapped=this._wrap;return jQuery.map(jQuery(jQuery.isArray(wrapped)?wrapped.join(""):wrapped).filter(filter||"*"),
function(e){return textOnly?e.innerText||e.textContent:e.outerHTML||outerHtml(e)})}
function tiUpdate(){var coll=this.nodes;jQuery.tmpl(null,null,null,this).insertBefore(coll[0]);jQuery(coll).remove()}})(jQuery);

/* - ++resource++plone.formwidget.recurrence/jquery.recurrenceinput.js - */
// https://www.gov.br/portal_javascripts/++resource++plone.formwidget.recurrence/jquery.recurrenceinput.js?original=1
"use strict";(function($){$.tools=$.tools||{version:'@VERSION'};var tool;var LABELS={};tool=$.tools.recurrenceinput={conf:{lang:'en',readOnly:false,firstDay:0,startField:null,startFieldYear:null,startFieldMonth:null,startFieldDay:null,ajaxURL:null,ajaxContentType:'application/json; charset=utf8',ributtonExtraClass:'',hasRepeatForeverButton:true,formOverlay:{speed:'fast',fixed:false},template:{form:'#jquery-recurrenceinput-form-tmpl',display:'#jquery-recurrenceinput-display-tmpl'},rtemplate:{daily:{rrule:'FREQ=DAILY',fields:['ridailyinterval','rirangeoptions']},mondayfriday:{rrule:'FREQ=WEEKLY;BYDAY=MO,FR',fields:['rirangeoptions']},weekdays:{rrule:'FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR',fields:['rirangeoptions']},weekly:{rrule:'FREQ=WEEKLY',fields:['riweeklyinterval','riweeklyweekdays','rirangeoptions']},monthly:{rrule:'FREQ=MONTHLY',fields:['rimonthlyinterval','rimonthlyoptions','rirangeoptions']},yearly:{rrule:'FREQ=YEARLY',fields:['riyearlyinterval','riyearlyoptions','rirangeoptions']}}},localize: function(language,labels){LABELS[language]=labels},setTemplates: function(templates,titles){var lang,template;tool.conf.rtemplate=templates;for(lang in titles){if(titles.hasOwnProperty(lang)){for(template in titles[lang]){if(titles[lang].hasOwnProperty(template)){LABELS[lang].rtemplate[template]=titles[lang][template]}}}}}};tool.localize("en",{displayUnactivate:'Does not repeat',displayActivate:'Repeats every',add_rules:'Add',edit_rules:'Edit',delete_rules:'Delete',add:'Add',refresh:'Refresh',title:'Repeat',preview:'Selected dates',addDate:'Add date',recurrenceType:'Repeats:',dailyInterval1:'Repeat every:',dailyInterval2:'days',weeklyInterval1:'Repeat every:',weeklyInterval2:'week(s)',weeklyWeekdays:'Repeat on:',weeklyWeekdaysHuman:'on:',monthlyInterval1:'Repeat every:',monthlyInterval2:'month(s)',monthlyDayOfMonth1:'Day',monthlyDayOfMonth1Human:'on day',monthlyDayOfMonth2:'of the month',monthlyDayOfMonth3:'month(s)',monthlyWeekdayOfMonth1:'The',monthlyWeekdayOfMonth1Human:'on the',monthlyWeekdayOfMonth2:'',monthlyWeekdayOfMonth3:'of the month',monthlyRepeatOn:'Repeat on:',yearlyInterval1:'Repeat every:',yearlyInterval2:'year(s)',yearlyDayOfMonth1:'Every',yearlyDayOfMonth1Human:'on',yearlyDayOfMonth2:'',yearlyDayOfMonth3:'',yearlyWeekdayOfMonth1:'The',yearlyWeekdayOfMonth1Human:'on the',yearlyWeekdayOfMonth2:'',yearlyWeekdayOfMonth3:'of',yearlyWeekdayOfMonth4:'',yearlyRepeatOn:'Repeat on:',range:'End recurrence:',rangeNoEnd:'Never',rangeByOccurrences1:'After',rangeByOccurrences1Human:'ends after',rangeByOccurrences2:'occurrence(s)',rangeByEndDate:'On',rangeByEndDateHuman:'ends on',including:', and also',except:', except for',cancel:'Cancel',save:'Save',recurrenceStart:'Start of the recurrence',additionalDate:'Additional date',include:'Include',exclude:'Exclude',remove:'Remove',orderIndexes:['first','second','third','fourth','last'],months:['January','February','March','April','May','June','July','August','September','October','November','December'],shortMonths:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],weekdays:['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],shortWeekdays:['Sun','Mon','Tue','Wed','Thu','Fri','Sat'],longDateFormat:'mmmm dd, yyyy',shortDateFormat:'mm/dd/yyyy',unsupportedFeatures:'Warning: This event uses recurrence features not '+'supported by this widget. Saving the recurrence '+'may change the recurrence in unintended ways:',noTemplateMatch:'No matching recurrence template',multipleDayOfMonth:'This widget does not support multiple days in monthly or yearly recurrence',bysetpos:'BYSETPOS is not supported',noRule:'No RRULE in RRULE data',noRepeatEvery:'Error: The "Repeat every"-field must be between 1 and 1000',noEndDate:'Error: End date is not set. Please set a correct value',noRepeatOn:'Error: "Repeat on"-value must be selected',pastEndDate:'Error: End date cannot be before start date',noEndAfterNOccurrences:'Error: The "After N occurrences"-field must be between 1 and 1000',alreadyAdded:'This date was already added',rtemplate:{daily:'Daily',mondayfriday:'Monday and Friday',weekdays:'Weekday',weekly:'Weekly',monthly:'Monthly',yearly:'Yearly'}});var OCCURRENCETMPL=['<div class="rioccurrences">','{{each occurrences}}','<div class="occurrence ${occurrences[$index].type}">','<span>','${occurrences[$index].formattedDate}','{{if occurrences[$index].type === "start"}}','<span class="rlabel">${i18n.recurrenceStart}</span>','{{/if}}','{{if occurrences[$index].type === "rdate"}}','<span class="rlabel">${i18n.additionalDate}</span>','{{/if}}','</span>','{{if !readOnly}}','<span class="action">','{{if occurrences[$index].type === "rrule"}}','<a date="${occurrences[$index].date}" href="#"','class="${occurrences[$index].type}" title="${i18n.exclude}">','${i18n.exclude}','</a>','{{/if}}','{{if occurrences[$index].type === "rdate"}}','<a date="${occurrences[$index].date}" href="#"','class="${occurrences[$index].type}" title="${i18n.remove}" >','${i18n.remove}','</a>','{{/if}}','{{if occurrences[$index].type === "exdate"}}','<a date="${occurrences[$index].date}" href="#"','class="${occurrences[$index].type}" title="${i18n.include}">','${i18n.include}','</a>','{{/if}}','</span>','{{/if}}','</div>','{{/each}}','<div class="batching">','{{each batch.batches}}','{{if $index === batch.currentBatch}}<span class="current">{{/if}}','<a href="#" start="${batch.batches[$index][0]}">[${batch.batches[$index][0]} - ${batch.batches[$index][1]}]</a>','{{if $index === batch.currentBatch}}</span>{{/if}}','{{/each}}','</div></div>'].join('\n');$.template('occurrenceTmpl',OCCURRENCETMPL);var DISPLAYTMPL=['<div class="ridisplay">','<div class="rimain">','{{if !readOnly}}','<a href="#" name="riedit">${i18n.add_rules}</a>','<a href="#" name="ridelete" style="display:none">${i18n.delete_rules}</a>','{{/if}}','<label class="ridisplay">${i18n.displayUnactivate}</label>','</div>','<div class="rioccurrences" style="display:none" /></div>'].join('\n');$.template('displayTmpl',DISPLAYTMPL);var FORMTMPL=['<div class="riform">','<form>','<h1>${i18n.title}</h1>','<div id="messagearea" style="display: none;">','</div>','<div id="rirtemplate">','<label for="${name}rtemplate" class="label">','${i18n.recurrenceType}','</label>','<select id="rirtemplate" name="rirtemplate" class="field">','{{each rtemplate}}','<option value="${$index}">${i18n.rtemplate[$index]}</value>','{{/each}}','</select>','<div>','<div id="riformfields">','<div id="ridailyinterval" class="rifield">','<label for="${name}dailyinterval" class="label">','${i18n.dailyInterval1}','</label>','<div class="field">','<input type="text" size="2"','value="1"','name="ridailyinterval"','id="${name}dailyinterval" />','${i18n.dailyInterval2}','</div>','</div>','<div id="riweeklyinterval" class="rifield">','<label for="${name}weeklyinterval" class="label">','${i18n.weeklyInterval1}','</label>','<div class="field">','<input type="text" size="2"','value="1"','name="riweeklyinterval"','id="${name}weeklyinterval"/>','${i18n.weeklyInterval2}','</div>','</div>','<div id="riweeklyweekdays" class="rifield">','<label for="${name}weeklyinterval" class="label">${i18n.weeklyWeekdays}</label>','<div class="field">','{{each orderedWeekdays}}','<div class="riweeklyweekday">','<input type="checkbox"','name="riweeklyweekdays${weekdays[$value]}"','id="${name}weeklyWeekdays${weekdays[$value]}"','value="${weekdays[$value]}" />','<label for="${name}weeklyWeekdays${weekdays[$value]}">${i18n.shortWeekdays[$value]}</label>','</div>','{{/each}}','</div>','</div>','<div id="rimonthlyinterval" class="rifield">','<label for="rimonthlyinterval" class="label">${i18n.monthlyInterval1}</label>','<div class="field">','<input type="text" size="2"','value="1" ','name="rimonthlyinterval"/>','${i18n.monthlyInterval2}','</div>','</div>','<div id="rimonthlyoptions" class="rifield">','<label for="rimonthlytype" class="label">${i18n.monthlyRepeatOn}</label>','<div class="field">','<div>','<input','type="radio"','value="DAYOFMONTH"','name="rimonthlytype"','id="${name}monthlytype:DAYOFMONTH" />','<label for="${name}monthlytype:DAYOFMONTH">','${i18n.monthlyDayOfMonth1}','</label>','<select name="rimonthlydayofmonthday"','id="${name}monthlydayofmonthday">','{{each [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,','18,19,20,21,22,23,24,25,26,27,28,29,30,31]}}','<option value="${$value}">${$value}</option>','{{/each}}','</select>','${i18n.monthlyDayOfMonth2}','</div>','<div>','<input','type="radio"','value="WEEKDAYOFMONTH"','name="rimonthlytype"','id="${name}monthlytype:WEEKDAYOFMONTH" />','<label for="${name}monthlytype:WEEKDAYOFMONTH">','${i18n.monthlyWeekdayOfMonth1}','</label>','<select name="rimonthlyweekdayofmonthindex">','{{each i18n.orderIndexes}}','<option value="${orderIndexes[$index]}">${$value}</option>','{{/each}}','</select>','${i18n.monthlyWeekdayOfMonth2}','<select name="rimonthlyweekdayofmonth">','{{each orderedWeekdays}}','<option value="${weekdays[$value]}">${i18n.weekdays[$value]}</option>','{{/each}}','</select>','${i18n.monthlyWeekdayOfMonth3}','</div>','</div>','</div>','<div id="riyearlyinterval" class="rifield">','<label for="riyearlyinterval" class="label">${i18n.yearlyInterval1}</label>','<div class="field">','<input type="text" size="2"','value="1" ','name="riyearlyinterval"/>','${i18n.yearlyInterval2}','</div>','</div>','<div id="riyearlyoptions" class="rifield">','<label for="riyearlyType" class="label">${i18n.yearlyRepeatOn}</label>','<div class="field">','<div>','<input','type="radio"','value="DAYOFMONTH"','name="riyearlyType"','id="${name}yearlytype:DAYOFMONTH" />','<label for="${name}yearlytype:DAYOFMONTH">','${i18n.yearlyDayOfMonth1}','</label>','<select name="riyearlydayofmonthmonth">','{{each i18n.months}}','<option value="${$index+1}">${$value}</option>','{{/each}}','</select>','${i18n.yearlyDayOfMonth2}','<select name="riyearlydayofmonthday">','{{each [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,','18,19,20,21,22,23,24,25,26,27,28,29,30,31]}}','<option value="${$value}">${$value}</option>','{{/each}}','</select>','${i18n.yearlyDayOfMonth3}','</div>','<div>','<input','type="radio"','value="WEEKDAYOFMONTH"','name="riyearlyType"','id="${name}yearlytype:WEEKDAYOFMONTH"/>','<label for="${name}yearlytype:WEEKDAYOFMONTH">','${i18n.yearlyWeekdayOfMonth1}','</label>','<select name="riyearlyweekdayofmonthindex">','{{each i18n.orderIndexes}}','<option value="${orderIndexes[$index]}">${$value}</option>','{{/each}}','</select>','<label for="${name}yearlytype:WEEKDAYOFMONTH">','${i18n.yearlyWeekdayOfMonth2}','</label>','<select name="riyearlyweekdayofmonthday">','{{each orderedWeekdays}}','<option value="${weekdays[$value]}">${i18n.weekdays[$value]}</option>','{{/each}}','</select>','${i18n.yearlyWeekdayOfMonth3}','<select name="riyearlyweekdayofmonthmonth">','{{each i18n.months}}','<option value="${$index+1}">${$value}</option>','{{/each}}','</select>','${i18n.yearlyWeekdayOfMonth4}','</div>','</div>','</div>','<div id="rirangeoptions" class="rifield">','<label class="label">${i18n.range}</label>','<div class="field">','{{if hasRepeatForeverButton}}','<div>','<input','type="radio"','value="NOENDDATE"','name="rirangetype"','id="${name}rangetype:NOENDDATE"/>','<label for="${name}rangetype:NOENDDATE">','${i18n.rangeNoEnd}','</label>','</div>','{{/if}}','<div>','<input','type="radio"','checked="checked"','value="BYOCCURRENCES"','name="rirangetype"','id="${name}rangetype:BYOCCURRENCES"/>','<label for="${name}rangetype:BYOCCURRENCES">','${i18n.rangeByOccurrences1}','</label>','<input','type="text" size="3"','value="7"','name="rirangebyoccurrencesvalue" />','${i18n.rangeByOccurrences2}','</div>','<div>','<input','type="radio"','value="BYENDDATE"','name="rirangetype"','id="${name}rangetype:BYENDDATE"/>','<label for="${name}rangetype:BYENDDATE">','${i18n.rangeByEndDate}','</label>','<input','type="date"','name="rirangebyenddatecalendar" />','</div>','</div>','</div>','</div>','<div class="rioccurrencesactions">','<div class="rioccurancesheader">','<h2>${i18n.preview}</h2>','<span class="refreshbutton action">','<a class="rirefreshbutton" href="#" title="${i18n.refresh}">','${i18n.refresh}','</a>','</span>','</div>','</div>','<div class="rioccurrences">','</div>','<div class="rioccurrencesactions">','<div class="rioccurancesheader">','<h2>${i18n.addDate}</h2>','</div>','<div class="riaddoccurrence">','<div class="errorarea"></div>','<input type="date" name="adddate" id="adddate" />','<input type="button" name="addaction" id="addaction" value="${i18n.add}">','</div>','</div>','<div class="ributtons">','<input','type="submit"','class="ricancelbutton ${ributtonExtraClass}"','value="${i18n.cancel}" />','<input','type="submit"','class="risavebutton ${ributtonExtraClass}"','value="${i18n.save}" />','</div>','</form></div>'].join('\n');$.template('formTmpl',FORMTMPL);var Re=/d{1,4}|m{1,4}|yy(?:yy)?|"[^"]*"|'[^']*'/g;
function zeropad(val,len){val=val.toString();len=len||2;while(val.length<len){val="0"+val}
return val}
function format(date,fmt,conf){var d=date.getDate(),D=date.getDay(),m=date.getMonth(),y=date.getFullYear(),flags={d:d,dd:zeropad(d),ddd:conf.i18n.shortWeekdays[D],dddd:conf.i18n.weekdays[D],m:m+1,mm:zeropad(m+1),mmm:conf.i18n.shortMonths[m],mmmm:conf.i18n.months[m],yy:String(y).slice(2),yyyy:y};var result=fmt.replace(Re, function($0){return flags.hasOwnProperty($0)?flags[$0]:$0.slice(1,$0.length-1)});return result}
function widgetSaveToRfc5545(form,conf,tz){var value=form.find('select[name=rirtemplate]').val();var rtemplate=conf.rtemplate[value];var result=rtemplate.rrule;var human=conf.i18n.rtemplate[value];var field,input,weekdays,i18nweekdays,i,j,index,tmp;var day,month,year,interval,yearlyType,occurrences,date;for(i=0;i<rtemplate.fields.length;i++){field=form.find('#'+rtemplate.fields[i]);switch(field.attr('id')){case 'ridailyinterval':interval=field.find('input[name=ridailyinterval]').val();if(interval!=='1'){result+=';INTERVAL='+interval}
human=interval+' '+conf.i18n.dailyInterval2;break;case 'riweeklyinterval':interval=field.find('input[name=riweeklyinterval]').val();if(interval!=='1'){result+=';INTERVAL='+interval}
human=interval+' '+conf.i18n.weeklyInterval2;break;case 'riweeklyweekdays':weekdays='';i18nweekdays='';for(j=0;j<conf.weekdays.length;j++){input=field.find('input[name=riweeklyweekdays'+conf.weekdays[j]+']');if(input.is(':checked')){if(weekdays){weekdays+=',';i18nweekdays+=', '}
weekdays+=conf.weekdays[j];i18nweekdays+=conf.i18n.weekdays[j]}}
if(weekdays){result+=';BYDAY='+weekdays;human+=' '+conf.i18n.weeklyWeekdaysHuman+' '+i18nweekdays}
break;case 'rimonthlyinterval':interval=field.find('input[name=rimonthlyinterval]').val();if(interval!=='1'){result+=';INTERVAL='+interval}
human=interval+' '+conf.i18n.monthlyInterval2;break;case 'rimonthlyoptions':var monthlyType=$('input[name=rimonthlytype]:checked',form).val();switch(monthlyType){case 'DAYOFMONTH':day=$('select[name=rimonthlydayofmonthday]',form).val();result+=';BYMONTHDAY='+day;human+=', '+conf.i18n.monthlyDayOfMonth1Human+' '+day+' '+conf.i18n.monthlyDayOfMonth2;break;case 'WEEKDAYOFMONTH':index=$('select[name=rimonthlyweekdayofmonthindex]',form).val();day=$('select[name=rimonthlyweekdayofmonth]',form).val();if($.inArray(day,['MO','TU','WE','TH','FR','SA','SU'])>-1){result+=';BYDAY='+index+day;human+=', '+conf.i18n.monthlyWeekdayOfMonth1Human+' ';human+=' '+conf.i18n.orderIndexes[$.inArray(index,conf.orderIndexes)];human+=' '+conf.i18n.monthlyWeekdayOfMonth2;human+=' '+conf.i18n.weekdays[$.inArray(day,conf.weekdays)];human+=' '+conf.i18n.monthlyDayOfMonth2}
break}
break;case 'riyearlyinterval':interval=field.find('input[name=riyearlyinterval]').val();if(interval!=='1'){result+=';INTERVAL='+interval}
human=interval+' '+conf.i18n.yearlyInterval2;break;case 'riyearlyoptions':yearlyType=$('input[name=riyearlyType]:checked',form).val();switch(yearlyType){case 'DAYOFMONTH':month=$('select[name=riyearlydayofmonthmonth]',form).val();day=$('select[name=riyearlydayofmonthday]',form).val();result+=';BYMONTH='+month;result+=';BYMONTHDAY='+day;human+=', '+conf.i18n.yearlyDayOfMonth1Human+' '+conf.i18n.months[month-1]+' '+day;break;case 'WEEKDAYOFMONTH':index=$('select[name=riyearlyweekdayofmonthindex]',form).val();day=$('select[name=riyearlyweekdayofmonthday]',form).val();month=$('select[name=riyearlyweekdayofmonthmonth]',form).val();result+=';BYMONTH='+month;if($.inArray(day,['MO','TU','WE','TH','FR','SA','SU'])>-1){result+=';BYDAY='+index+day;human+=', '+conf.i18n.yearlyWeekdayOfMonth1Human;human+=' '+conf.i18n.orderIndexes[$.inArray(index,conf.orderIndexes)];human+=' '+conf.i18n.yearlyWeekdayOfMonth2;human+=' '+conf.i18n.weekdays[$.inArray(day,conf.weekdays)];human+=' '+conf.i18n.yearlyWeekdayOfMonth3;human+=' '+conf.i18n.months[month-1];human+=' '+conf.i18n.yearlyWeekdayOfMonth4}
break}
break;case 'rirangeoptions':var rangeType=form.find('input[name=rirangetype]:checked').val();switch(rangeType){case 'BYOCCURRENCES':occurrences=form.find('input[name=rirangebyoccurrencesvalue]').val();result+=';COUNT='+occurrences;human+=', '+conf.i18n.rangeByOccurrences1Human;human+=' '+occurrences;human+=' '+conf.i18n.rangeByOccurrences2;break;case 'BYENDDATE':field=form.find('input[name=rirangebyenddatecalendar]');date=field.data('dateinput').getValue('yyyymmdd');result+=';UNTIL='+date+'T000000';if(tz===true){result+='Z'}
human+=', '+conf.i18n.rangeByEndDateHuman;human+=' '+field.data('dateinput').getValue(conf.i18n.longDateFormat);break}
break}}
if(form.ical.RDATE!==undefined&&form.ical.RDATE.length>0){form.ical.RDATE.sort();tmp=[];for(i=0;i<form.ical.RDATE.length;i++){if(form.ical.RDATE[i]!==''){year=parseInt(form.ical.RDATE[i].substring(0,4),10);month=parseInt(form.ical.RDATE[i].substring(4,6),10)-1;day=parseInt(form.ical.RDATE[i].substring(6,8),10);tmp.push(format(new Date(year,month,day),conf.i18n.longDateFormat,conf))}}
if(tmp.length!==0){human=human+conf.i18n.including+' '+tmp.join('; ')}}
if(form.ical.EXDATE!==undefined&&form.ical.EXDATE.length>0){form.ical.EXDATE.sort();tmp=[];for(i=0;i<form.ical.EXDATE.length;i++){if(form.ical.EXDATE[i]!==''){year=parseInt(form.ical.EXDATE[i].substring(0,4),10);month=parseInt(form.ical.EXDATE[i].substring(4,6),10)-1;day=parseInt(form.ical.EXDATE[i].substring(6,8),10);tmp.push(format(new Date(year,month,day),conf.i18n.longDateFormat,conf))}}
if(tmp.length!==0){human=human+conf.i18n.except+' '+tmp.join('; ')}}
result='RRULE:'+result;if(form.ical.EXDATE!==undefined&&form.ical.EXDATE.join()!==""){tmp=$.map(form.ical.EXDATE, function(x){if(x.length===8){x+='T000000'}
if(tz===true){x+='Z'}
return x});result=result+'\nEXDATE:'+tmp}
if(form.ical.RDATE!==undefined&&form.ical.RDATE.join()!==""){tmp=$.map(form.ical.RDATE, function(x){if(x.length===8){x+='T000000'}
if(tz===true){x+='Z'}
return x});result=result+'\nRDATE:'+tmp}
return{result:result,description:human}}
function parseLine(icalline){var result={};var pos=icalline.indexOf(':');var property=icalline.substring(0,pos);result.value=icalline.substring(pos+1);if(property.indexOf(';')!==-1){pos=property.indexOf(';');result.parameters=property.substring(pos+1);result.property=property.substring(0,pos)} else{result.parameters=null;result.property=property}
return result}
function cleanDates(dates){var result=[];var splitDates=dates.split(',');var date;for(date in splitDates){if(splitDates.hasOwnProperty(date)){if(splitDates[date].indexOf('Z')!==-1){result.push(splitDates[date].substring(0,15))} else{result.push(splitDates[date])}}}
return result}
function parseIcal(icaldata){var lines=[];var result={};var propAndValue=[];var line=null;var nextline;lines=icaldata.split('\n');lines.reverse();while(true){if(lines.length>0){nextline=lines.pop();if(nextline.charAt(0)===' '||nextline.charAt(0)==='\t'){line=line+nextline;continue}} else{nextline=''}
if(line!==null){line=parseLine(line);if(line.property==='RDATE'||line.property==='EXDATE'){result[line.property]=cleanDates(line.value)} else{result[line.property]=line.value}}
line=nextline;if(line===''){break}}
return result}
function widgetLoadFromRfc5545(form,conf,icaldata,force){var unsupportedFeatures=[];var i,matches,match,matchIndex,rtemplate,d,input,index;var selector,selectors,field,radiobutton,start,end;var interval,byday,bymonth,bymonthday,count,until;var day,month,year,weekday,ical;form.ical=parseIcal(icaldata);if(form.ical.RRULE===undefined){unsupportedFeatures.push(conf.i18n.noRule);if(!force){return-1}} else{matches=/INTERVAL=([0-9]+);?/.exec(form.ical.RRULE);if(matches){interval=matches[1]} else{interval='1'}
matches=/BYDAY=([^;]+);?/.exec(form.ical.RRULE);if(matches){byday=matches[1]} else{byday=''}
matches=/BYMONTHDAY=([^;]+);?/.exec(form.ical.RRULE);if(matches){bymonthday=matches[1].split(",")} else{bymonthday=null}
matches=/BYMONTH=([^;]+);?/.exec(form.ical.RRULE);if(matches){bymonth=matches[1].split(",")} else{bymonth=null}
matches=/COUNT=([0-9]+);?/.exec(form.ical.RRULE);if(matches){count=matches[1]} else{count=null}
matches=/UNTIL=([0-9T]+);?/.exec(form.ical.RRULE);if(matches){until=matches[1]} else{until=null}
matches=/BYSETPOS=([^;]+);?/.exec(form.ical.RRULE);if(matches){unsupportedFeatures.push(conf.i18n.bysetpos)}
match='';matchIndex=null;for(i in conf.rtemplate){if(conf.rtemplate.hasOwnProperty(i)){rtemplate=conf.rtemplate[i];if(form.ical.RRULE.indexOf(rtemplate.rrule)===0){if(form.ical.RRULE.length>match.length){match=form.ical.RRULE;matchIndex=i}}}}
if(match){rtemplate=conf.rtemplate[matchIndex];selector=form.find('select[name=rirtemplate]').val(matchIndex)} else{for(rtemplate in conf.rtemplate){if(conf.rtemplate.hasOwnProperty(rtemplate)){rtemplate=conf.rtemplate[rtemplate];break}}
unsupportedFeatures.push(conf.i18n.noTemplateMatch)}
for(i=0;i<rtemplate.fields.length;i++){field=form.find('#'+rtemplate.fields[i]);switch(field.attr('id')){case 'ridailyinterval':field.find('input[name=ridailyinterval]').val(interval);break;case 'riweeklyinterval':field.find('input[name=riweeklyinterval]').val(interval);break;case 'riweeklyweekdays':byday=byday.split(",");for(d=0;d<conf.weekdays.length;d++){day=conf.weekdays[d];input=field.find('input[name=riweeklyweekdays'+day+']');input.attr('checked',$.inArray(day,byday)!==-1)}
break;case 'rimonthlyinterval':field.find('input[name=rimonthlyinterval]').val(interval);break;case 'rimonthlyoptions':var monthlyType='DAYOFMONTH';if(bymonthday){monthlyType='DAYOFMONTH';if(bymonthday.length>1){unsupportedFeatures.push(conf.i18n.multipleDayOfMonth);bymonthday=bymonthday[0]}
field.find('select[name=rimonthlydayofmonthday]').val(bymonthday)}
if(byday){monthlyType='WEEKDAYOFMONTH';if(byday.indexOf(',')!==-1){unsupportedFeatures.push(conf.i18n.multipleDayOfMonth);byday=byday.split(",")[0]}
index=byday.slice(0,-2);if(index.charAt(0)!=='+'&&index.charAt(0)!=='-'){index='+'+index}
weekday=byday.slice(-2);field.find('select[name=rimonthlyweekdayofmonthindex]').val(index);field.find('select[name=rimonthlyweekdayofmonth]').val(weekday)}
selectors=field.find('input[name=rimonthlytype]');for(index=0;index<selectors.length;index++){radiobutton=selectors[index];$(radiobutton).attr('checked',radiobutton.value===monthlyType)}
break;case 'riyearlyinterval':field.find('input[name=riyearlyinterval]').val(interval);break;case 'riyearlyoptions':var yearlyType='DAYOFMONTH';if(bymonthday){yearlyType='DAYOFMONTH';if(bymonthday.length>1){unsupportedFeatures.push(conf.i18n.multipleDayOfMonth);bymonthday=bymonthday[0]}
field.find('select[name=riyearlydayofmonthmonth]').val(bymonth);field.find('select[name=riyearlydayofmonthday]').val(bymonthday)}
if(byday){yearlyType='WEEKDAYOFMONTH';if(byday.indexOf(',')!==-1){unsupportedFeatures.push(conf.i18n.multipleDayOfMonth);byday=byday.split(",")[0]}
index=byday.slice(0,-2);if(index.charAt(0)!=='+'&&index.charAt(0)!=='-'){index='+'+index}
weekday=byday.slice(-2);field.find('select[name=riyearlyweekdayofmonthindex]').val(index);field.find('select[name=riyearlyweekdayofmonthday]').val(weekday);field.find('select[name=riyearlyweekdayofmonthmonth]').val(bymonth)}
selectors=field.find('input[name=riyearlyType]');for(index=0;index<selectors.length;index++){radiobutton=selectors[index];$(radiobutton).attr('checked',radiobutton.value===yearlyType)}
break;case 'rirangeoptions':var rangeType='NOENDDATE';if(count){rangeType='BYOCCURRENCES';field.find('input[name=rirangebyoccurrencesvalue]').val(count)}
if(until){rangeType='BYENDDATE';input=field.find('input[name=rirangebyenddatecalendar]');year=until.slice(0,4);month=until.slice(4,6);month=parseInt(month,10)-1;day=until.slice(6,8);input.data('dateinput').setValue(new Date(year,month,day))}
selectors=field.find('input[name=rirangetype]');for(index=0;index<selectors.length;index++){radiobutton=selectors[index];$(radiobutton).attr('checked',radiobutton.value===rangeType)}
break}}}
var messagearea=form.find('#messagearea');if(unsupportedFeatures.length!==0){messagearea.text(conf.i18n.unsupportedFeatures+' '+unsupportedFeatures.join('; '));messagearea.show();return 1} else{messagearea.text('');messagearea.hide();return 0}}
function RecurrenceInput(conf,textarea){var self=this;var form,display;var orderedWeekdays=[];var index,i;for(i=0;i<7;i++){index=i+conf.firstDay;if(index>6){index=index-7}
orderedWeekdays.push(index)}
$.extend(conf,{orderIndexes:['+1','+2','+3','+4','-1'],weekdays:['SU','MO','TU','WE','TH','FR','SA'],orderedWeekdays:orderedWeekdays});
function displayFields(selector){var i;form.find('.rifield').hide();var value=selector.val();if(value){var rtemplate=conf.rtemplate[value];for(i=0;i<rtemplate.fields.length;i++){form.find('#'+rtemplate.fields[i]).show()}}}
function occurrenceExclude(event){event.preventDefault();if(form.ical.EXDATE===undefined){form.ical.EXDATE=[]}
form.ical.EXDATE.push(this.attributes.date.value);var $this=$(this);$this.attr('class','exdate');$this.parent().parent().addClass('exdate');$this.unbind(event);$this.click(occurrenceInclude)}
function occurrenceInclude(event){event.preventDefault();form.ical.EXDATE.splice($.inArray(this.attributes.date.value,form.ical.EXDATE),1);var $this=$(this);$this.attr('class','rrule');$this.parent().parent().removeClass('exdate');$this.unbind(event);$this.click(occurrenceExclude)}
function occurrenceDelete(event){event.preventDefault();form.ical.RDATE.splice($.inArray(this.attributes.date.value,form.ical.RDATE),1);$(this).parent().parent().hide('slow', function(){$(this).remove()})}
function occurrenceAdd(event){event.preventDefault();var dateinput=form.find('.riaddoccurrence input#adddate').data('dateinput');var datevalue=dateinput.getValue('yyyymmddT000000');if(form.ical.RDATE===undefined){form.ical.RDATE=[]}
var errorarea=form.find('.riaddoccurrence div.errorarea');errorarea.text('');errorarea.hide();if($.inArray(datevalue,form.ical.RDATE)===-1){form.ical.RDATE.push(datevalue);var html=['<div class="occurrence rdate" style="display: none;">','<span class="rdate">',dateinput.getValue(conf.i18n.longDateFormat),'<span class="rlabel">'+conf.i18n.additionalDate+'</span>','</span>','<span class="action">','<a date="'+datevalue+'" href="#" class="rdate" >','Include','</a>','</span>','</div>'].join('\n');form.find('div.rioccurrences').prepend(html);$(form.find('div.rioccurrences div')[0]).slideDown();$(form.find('div.rioccurrences .action a.rdate')[0]).click(occurrenceDelete)} else{errorarea.text(conf.i18n.alreadyAdded).show()}}
function loadOccurrences(startdate,rfc5545,start,readonly){var element,occurrenceDiv;if(!readonly){element=form} else{element=display}
occurrenceDiv=element.find('.rioccurrences');occurrenceDiv.hide();var year,month,day;year=startdate.getFullYear();month=startdate.getMonth()+1;day=startdate.getDate();var data={year:year,month:month,day:day,rrule:rfc5545,format:conf.i18n.longDateFormat,start:start};var dict={url:conf.ajaxURL,async:false,type:'post',dataType:'json',contentType:conf.ajaxContentType,cache:false,data:data,success: function(data,status,jqXHR){var result,element;if(!readonly){element=form} else{element=display}
data.readOnly=readonly;data.i18n=conf.i18n;var occurrence,date,y,m,d,each;for(each in data.occurrences){if(data.occurrences.hasOwnProperty(each)){occurrence=data.occurrences[each];date=occurrence.date;y=parseInt(date.substring(0,4),10);m=parseInt(date.substring(4,6),10)-1;d=parseInt(date.substring(6,8),10);occurrence.formattedDate=format(new Date(y,m,d),conf.i18n.longDateFormat,conf)}}
result=$.tmpl('occurrenceTmpl',data);occurrenceDiv=element.find('.rioccurrences');occurrenceDiv.replaceWith(result);element.find('.rioccurrences .batching a').click(
function(event){event.preventDefault();loadOccurrences(startdate,rfc5545,this.attributes.start.value,readonly)});if(!readonly){element.find('.rioccurrences .action a.rrule').click(occurrenceExclude);element.find('.rioccurrences .action a.exdate').click(occurrenceInclude);element.find('.rioccurrences .action a.rdate').click(occurrenceDelete)}
element.find('.rioccurrences').show()},error: function(jqXHR,textStatus,errorThrown){alert(textStatus)}};$.ajax(dict)}
function getField(field){var realField=$(field);if(!realField.length){realField=$('#'+field)}
if(!realField.length){realField=$("input[name='"+field+"']")}
return realField}
function findStartDate(){var startdate=null;var startField,startFieldYear,startFieldMonth,startFieldDay;if(conf.startField){startField=getField(conf.startField);if(!startField.length){return null}
startdate=startField.data('dateinput');if(!startdate){startdate=startField.val();if(startdate===""){startdate=startField.text()}} else{startdate=startdate.getValue()}
if(typeof startdate==='string'){startdate=startdate.replace(' ','T')}
startdate=new Date(startdate)} else if(conf.startFieldYear&&conf.startFieldMonth&&conf.startFieldDay){startFieldYear=getField(conf.startFieldYear);startFieldMonth=getField(conf.startFieldMonth);startFieldDay=getField(conf.startFieldDay);if(!startFieldYear.length&&!startFieldMonth.length&&!startFieldDay.length){return null}
startdate=new Date(startFieldYear.val(),startFieldMonth.val()-1,startFieldDay.val())}
if(startdate===null){return null}
if(isNaN(startdate)){return null}
return startdate}
function findEndDate(form){var endField,enddate;endField=form.find('input[name=rirangebyenddatecalendar]');enddate=endField.data('dateinput');if(!enddate){enddate=endField.val()} else{enddate=enddate.getValue()}
enddate=new Date(enddate);if(isNaN(enddate)||endField.val()===""){return null}
return enddate}
function findIntField(fieldName,form){var field,num,isInt;field=form.find('input[name='+fieldName+']');num=field.val();if(isNaN(num)||(num.toString().indexOf('.')!==-1)||field.val()===""){return null}
return num}
function loadData(rfc5545){var selector,format,startdate,dayindex,day;if(rfc5545){widgetLoadFromRfc5545(form,conf,rfc5545,true)}
startdate=findStartDate();if(startdate!==null){form.find('select[name=rimonthlydayofmonthday]').val(startdate.getDate());dayindex=conf.orderIndexes[Math.floor((startdate.getDate()-1)/7)];day=conf.weekdays[startdate.getDay()];form.find('select[name=rimonthlyweekdayofmonthindex]').val(dayindex);form.find('select[name=rimonthlyweekdayofmonth]').val(day);form.find('select[name=riyearlydayofmonthmonth]').val(startdate.getMonth()+1);form.find('select[name=riyearlydayofmonthday]').val(startdate.getDate());form.find('select[name=riyearlyweekdayofmonthindex]').val(dayindex);form.find('select[name=riyearlyweekdayofmonthday]').val(day);form.find('select[name=riyearlyweekdayofmonthmonth]').val(startdate.getMonth()+1);loadOccurrences(startdate,widgetSaveToRfc5545(form,conf,false).result,0,false);form.find('div.rioccurrencesactions').show()} else{form.find('div.rioccurrencesactions').hide()}
selector=form.find('select[name=rirtemplate]');displayFields(selector)}
function recurrenceOn(){var RFC5545=widgetSaveToRfc5545(form,conf,false);var label=display.find('label[class=ridisplay]');label.text(conf.i18n.displayActivate+' '+RFC5545.description);textarea.val(RFC5545.result).change();var startdate=findStartDate();if(startdate!==null){loadOccurrences(startdate,widgetSaveToRfc5545(form,conf,false).result,0,true)}
display.find('a[name="riedit"]').text(conf.i18n.edit_rules);display.find('a[name="ridelete"]').show()}
function recurrenceOff(){var label=display.find('label[class=ridisplay]');label.text(conf.i18n.displayUnactivate);textarea.val('').change();display.find('.rioccurrences').hide();display.find('a[name="riedit"]').text(conf.i18n.add_rules);display.find('a[name="ridelete"]').hide()}
function checkFields(form){var startDate,endDate,num,messagearea;startDate=findStartDate();messagearea=form.find('#messagearea');messagearea.text('');messagearea.hide();form.find('.riaddoccurrence div.errorarea').text('').hide();if(form.find('#ridailyinterval').css('display')==='block'){num=findIntField('ridailyinterval',form);if(!num||num<1||num>1000){messagearea.text(conf.i18n.noRepeatEvery).show();return false}}
if(form.find('#riweeklyinterval').css('display')==='block'){num=findIntField('riweeklyinterval',form);if(!num||num<1||num>1000){messagearea.text(conf.i18n.noRepeatEvery).show();return false}}
if(form.find('#rimonthlyinterval').css('display')==='block'){num=findIntField('rimonthlyinterval',form);if(!num||num<1||num>1000){messagearea.text(conf.i18n.noRepeatEvery).show();return false}
if(form.find('#rimonthlyoptions input:checked').length===0){messagearea.text(conf.i18n.noRepeatOn).show();return false}}
if(form.find('#riyearlyinterval').css('display')==='block'){num=findIntField('riyearlyinterval',form);if(!num||num<1||num>1000){messagearea.text(conf.i18n.noRepeatEvery).show();return false}
if(form.find('#riyearlyoptions input:checked').length===0){messagearea.text(conf.i18n.noRepeatOn).show();return false}}
if(form.find('input[value="BYOCCURRENCES"]:visible:checked').length>0){num=findIntField('rirangebyoccurrencesvalue',form);if(!num||num<1||num>1000){messagearea.text(conf.i18n.noEndAfterNOccurrences).show();return false}}
if(form.find('input[value="BYENDDATE"]:visible:checked').length>0){endDate=findEndDate(form);if(!endDate){messagearea.text(conf.i18n.noEndDate).show();return false} else if(endDate<startDate){messagearea.text(conf.i18n.pastEndDate).show();return false}}
return true}
function save(event){event.preventDefault();if(checkFields(form)){form.overlay().close();recurrenceOn()}}
function cancel(event){event.preventDefault();form.overlay().close()}
function updateOccurances(){var startDate;startDate=findStartDate();if(checkFields(form)){loadOccurrences(startDate,widgetSaveToRfc5545(form,conf,false).result,0,false)}}
display=$.tmpl('displayTmpl',conf);form=$.tmpl('formTmpl',conf);form.overlay(conf.formOverlay).hide();form.ical={RDATE:[],EXDATE:[]};$.tools.dateinput.localize(conf.lang,{months:LABELS[conf.lang].months.join(),shortMonths:LABELS[conf.lang].shortMonths.join(),days:LABELS[conf.lang].weekdays.join(),shortDays:LABELS[conf.lang].shortWeekdays.join()});form.find('input[name=rirangebyenddatecalendar]').dateinput({selectors:true,lang:conf.lang,format:conf.i18n.shortDateFormat,firstDay:conf.firstDay,yearRange:[-5,10]}).data('dateinput').setValue(new Date());if(textarea.val()){var result=widgetLoadFromRfc5545(form,conf,textarea.val(),false);if(result===-1){var label=display.find('label[class=ridisplay]');label.text(conf.i18n.noRule)} else{recurrenceOn()}}
display.find('a[name=ridelete]').click(function(e){e.preventDefault();recurrenceOff()});display.find('a[name=riedit]').click(
function(e){loadData(textarea.val());e.preventDefault();form.overlay().load()});form.find('div.riaddoccurrence input#adddate').dateinput({selectors:true,lang:conf.lang,format:conf.i18n.shortDateFormat,firstDay:conf.firstDay,yearRange:[-5,10]}).data('dateinput').setValue(new Date());form.find('input#addaction').click(occurrenceAdd);form.find('a.rirefreshbutton').click(
function(event){event.preventDefault();updateOccurances()});form.find('select[name=rirtemplate]').change(
function(e){displayFields($(this))});form.find('select').change(
function(e){$(this).parent().find('> input').click().change()});form.find('input[name=rirangebyoccurrencesvalue]').change(
function(e){$(this).parent().find('input[name=rirangetype]').click().change()});form.find('input[name=rirangebyenddatecalendar]').change(function(){$(this).parent().find('input[name=rirangetype]').click();if(form.find('.rioccurrencesactions:visible').length!==0){updateOccurances()}});form.find('input:radio, .riweeklyweekday > input, input[name=ridailyinterval], input[name=riweeklyinterval], input[name=rimonthlyinterval], input[name=riyearlyinterval]').change(
function(e){if(form.find('.rioccurrencesactions:visible').length!==0){updateOccurances()}});form.find('.ricancelbutton').click(cancel);form.find('.risavebutton').click(save);$.extend(self,{display:display,form:form,loadData:loadData,save:save})}
$.fn.recurrenceinput=function(conf){if(this.data('recurrenceinput')){return this.data('recurrenceinput')}
var config=$.extend({},tool.conf);$.extend(config,conf);$.extend(config,{i18n:LABELS[config.lang],name:this.attr('name')});var recurrenceinput=new RecurrenceInput(config,this);recurrenceinput.form.appendTo('body');this.after(recurrenceinput.display);this.hide();this.data('recurrenceinput',recurrenceinput);return recurrenceinput}}(jQuery));

/* - ++resource++plone.app.event/event.js - */
// https://www.gov.br/portal_javascripts/++resource++plone.app.event/event.js?original=1
if(typeof(plone)==='undefined'){var plone={}}(function($,plone){plone.paevent=plone.paevent||{};plone.paevent.end_start_delta=1;plone.paevent.start_date=function(){return a_or_b($('#formfield-form-widgets-IEventBasic-start'),$('#archetypes-fieldname-startDate'))};plone.paevent.end_date=function(){return a_or_b($('#formfield-form-widgets-IEventBasic-end'),$('#archetypes-fieldname-endDate'))};
function initDelta(e){var start_datetime=getDateTime(plone.paevent.start_date());var end_datetime=getDateTime(plone.paevent.end_date());plone.paevent.end_start_delta=(end_datetime-start_datetime)/ 1000 / 3600;if(plone.paevent.end_start_delta<1){plone.paevent.end_start_delta=1}}
function getDateTime(datetimewidget){var fields=['year','month','day','hour','minute'];var parts={};$.each(fields, function(idx,fld){parts[fld]=parseInt($("[name$='-"+fld+"']",datetimewidget).val(),10)});var date=new Date(parts.year,parts.month-1,parts.day,parts.hour,parts.minute);return date}
function setDateTime(datetimewidget,new_date){$("[name$='-year']",datetimewidget).val(new_date.getYear());$("[name$='-month']",datetimewidget).val(new_date.getMonth());$("[name$='-day']",datetimewidget).val(new_date.getDate());$("[name$='-hour']",datetimewidget).val(new_date.getHours());$("[name$='-minute']",datetimewidget).val(new_date.getMinutes())}
function updateEndDate(){var start_date=getDateTime(plone.paevent.start_date());var new_end_date=start_date;new_end_date.setHours(new_end_date.getHours()+plone.paevent.end_start_delta);setDateTime(plone.paevent.end_date(),new_end_date);$("input[name$='-calendar']",plone.paevent.end_date()).data('dateinput').setValue(new_end_date)}
function validateEndDate(){var start_datetime=getDateTime(plone.paevent.start_date());var end_datetime=getDateTime(plone.paevent.end_date());if(end_datetime<start_datetime){plone.paevent.end_date().addClass("error")} else{plone.paevent.end_date().removeClass("error")}}
function show_hide_widget(widget,hide,fade){if(hide===true){if(fade===true){widget.fadeOut()}
else{widget.hide()}} else{if(fade===true){widget.fadeIn()}
else{widget.show()}}}
function a_or_b(a,b){var ret;if(a.length>0){ret=a} else{ret=b}
return ret}
function event_listing_calendar_init(cal){if($().dateinput&&cal.length>0){var get_req_param=function(name){if(name===(new RegExp('[?&]'+encodeURIComponent(name)+'=([^&]*)')).exec(location.search)){return decodeURIComponent(name[1])}};var val=get_req_param('date');if(val===undefined){val=new Date()} else{val=new Date(val)}
cal.dateinput({selectors:true,trigger:true,format:'yyyy-mm-dd',yearRange:[-10,10],firstDay:1,value:val,change: function(){var value=this.getValue("yyyy-mm-dd");window.location.href='event_listing?mode=day&date='+value}}).unbind('change').bind('onShow', function(event){var trigger_offset=$(this).next().offset();$(this).data('dateinput').getCalendar().offset({top:trigger_offset.top+20,left:trigger_offset.left})})}}
$(function(){var jq_whole_day=a_or_b($('#formfield-form-widgets-IEventBasic-whole_day input'),$('form[name="edit_form"] input#wholeDay'));var jq_datetime=a_or_b($('#form-widgets-IEventBasic-start-timecomponents, #form-widgets-IEventBasic-end-timecomponents'),$('#archetypes-fieldname-startDate .datetimewidget-time, #archetypes-fieldname-endDate .datetimewidget-time'));if(jq_whole_day.length>0){jq_whole_day.bind('change', function(e){show_hide_widget(jq_datetime,e.target.checked,true)});show_hide_widget(jq_datetime,jq_whole_day.get(0).checked,false)}
var jq_open_end=a_or_b($('#formfield-form-widgets-IEventBasic-open_end input'),$('form[name="edit_form"] input#openEnd'));var jq_end=plone.paevent.end_date();if(jq_open_end.length>0){jq_open_end.bind('change', function(e){show_hide_widget(jq_end,e.target.checked,true)});show_hide_widget(jq_end,jq_open_end.get(0).checked,false)}
var jq_start=plone.paevent.start_date();$("select",jq_start).on("change",updateEndDate);$("input[name$='-calendar']",jq_start).on("onHide",updateEndDate);$("select",jq_end).on("change", function(){initDelta();validateEndDate()});$("input[name$='-calendar']",jq_end).on("onHide", function(){initDelta();validateEndDate()});event_listing_calendar_init($("#event_listing_calendar"))})}(jQuery,plone));

/* - ++resource++plone.app.event.portlet_calendar.js - */
// https://www.gov.br/portal_javascripts/++resource++plone.app.event.portlet_calendar.js?original=1
(function($){
function load_portlet_calendar(event,elem){event.preventDefault();var pw=elem.closest('.portletWrapper');var elem_data=elem.data();var portlethash=pw.attr('id');portlethash=portlethash.substring(15,portlethash.length);url=portal_url+'/@@render-portlet?portlethash='+portlethash+'&year='+elem_data.year+'&month='+elem_data.month;$.ajax({url:url,success: function(data){pw.html(data);rebind_portlet_calendar()}})}
function rebind_portlet_calendar(){$('.portletCalendar a.calendarNext').click(function(event){load_portlet_calendar(event,$(this))});$('.portletCalendar a.calendarPrevious').click(function(event){load_portlet_calendar(event,$(this))});$('.portletCalendar dd a[title]').tooltip({offset:[-10,0],tipClass:'pae_calendar_tooltip'})}
$(document).ready(function(){rebind_portlet_calendar()})}(jQuery));

/* - ++resource++plone.formwidget.contenttree/contenttree.js - */
// This is based on jQueryFileTree by   Cory S.N. LaViska
if(jQuery) (function($){

    $.extend($.fn, {
        showDialog: function() {
            $(document.body)
                .append($(document.createElement("div"))
                .addClass("contenttreeWindowBlocker"));
            // store old parent element
            this[0].oldparent = $(this).parent()[0];
            $(".contenttreeWindowBlocker").before(this);
            $(this).show();
            $(this).width($(window).width() * 0.75);
            $(this).height($(window).height() * 0.75);
            $(this).css({
                'left': $(window).width() * 0.125,
                'top': $(window).height() * 0.125
            });
        },

        contentTreeAdd: function() {
            var contenttree_window = (this).parents(".contenttreeWindow");
            var input_box = $('#'+ contenttree_window[0].id
                .replace(/-contenttree-window$/,"-widgets-query"));
            contenttree_window.find('.navTreeCurrentItem > a').each(function () {
                formwidget_autocomplete_new_value(
                    input_box,
                    $(this).attr('href'),
                    $.trim($(this).text())
                );
            });
            $(this).contentTreeCancel();
            input_box.parents('.datagridwidget-cell').triggerHandler('change');
        },

        contentTreeCancel: function() {
            $(".contenttreeWindowBlocker").remove();
            var popup = $(this).parents(".contenttreeWindow");
            popup.hide();
            $(popup[0].oldparent).append(popup);
            popup[0].oldparent = null;
        },

        contentTree: function(o, h) {
            // Defaults
            if (!o) {
                var o = {};
            }
            if (o.script == undefined) {
                o.script = 'fetch';
            }

            if (o.folderEvent == undefined) {
                o.folderEvent = 'click';
            }
            if (o.selectEvent == undefined) {
                o.selectEvent = 'click';
            }

            if (o.expandSpeed == undefined) {
                o.expandSpeed = -1;
            }
            if (o.collapseSpeed == undefined) {
                o.collapseSpeed = -1;
            }

            if (o.multiFolder == undefined) {
                o.multiFolder = true;
            }
            if (o.multiSelect == undefined) {
                o.multiSelect = false;
            }

            function loadTree(c, t, r) {
                $(c).addClass('wait');
                $.post(o.script, {href: t, rel: r}, function(data) {
                    $(c).removeClass('wait').append(data);
                    $(c).find('ul:hidden').slideDown({
                        duration: o.expandSpeed
                    });
                    bindTree(c);
                });
            }

            function handleFolderEvent() {
                var li = $(this).parent();
                if (li.hasClass('collapsed')) {
                    if (!o.multiFolder) {
                        li.parent().find('ul:visible').slideUp({
                            duration: o.collapseSpeed
                        });
                        li.parent().find('li.navTreeFolderish')
                            .removeClass('expanded')
                            .addClass('collapsed');
                    }

                    if (li.find('ul').length == 0)
                        loadTree(
                            li,
                            escape($(this).attr('href')),
                            escape($(this).attr('rel'))
                        );
                    else
                        li.find('ul:hidden').slideDown({
                            duration: o.expandSpeed
                        });
                    li.removeClass('collapsed').addClass('expanded');
                } else {
                    li.find('ul').slideUp({
                        duration: o.collapseSpeed
                    });
                    li.removeClass('expanded').addClass('collapsed');
                }
                return false;
            }

            function handleSelectEvent(event) {
                var li = $(this).parent();
                var selected = true;
                var root = $(this).parents('ul.navTree');
                if (!li.hasClass('navTreeCurrentItem')) {
                    var multi_key = (
                        (event.ctrlKey) ||
                        (navigator.userAgent.toLowerCase()
                            .indexOf('macintosh') != -1 && event.metaKey));

                    if (!o.multiSelect || !multi_key) {
                        root.find('li.navTreeCurrentItem')
                            .removeClass('navTreeCurrentItem');
                    }

                    li.addClass('navTreeCurrentItem');
                    selected = true;
                } else {
                    li.removeClass('navTreeCurrentItem');
                    selected = false;
                }

                h(event, true, $(this).attr('href'), $.trim($(this).text()));
            }

            function bindTree(t) {
                $(t).find('li.navTreeFolderish a').unbind(o.folderEvent);
                $(t).find('li.selectable a').unbind(o.selectEvent);
                $(t).find('li a').bind('click', function() {
                    return false;
                });
                $(t).find('li.navTreeFolderish a').bind(
                    o.folderEvent,
                    handleFolderEvent
                );
                $(t).find('li.selectable a').bind(
                    o.selectEvent,
                    handleSelectEvent
                );
            }

            if ($(this).children('ul.navTree').length <= 0) {
                $(this).each(function() {
                    loadTree(this, o.rootUrl, 0);
                });
            }

        }
    });

})(jQuery);


/* - ++resource++plone.formwidget.querystring.querywidget.js - */
(function ($) {

    // Define querywidget namespace if it doesn't exist
    if (typeof($.querywidget) === "undefined") {
        $.querywidget = {
            config: {},
            initialized: false
        };
    }

    // Create a select menu
    $.querywidget.createSelect = function (values, selectedvalue, className, name) {

        // Create select
        var select = $(document.createElement('select'))
                            .addClass(className)
                            .attr('name', name);
        $.each(values, function (i, val) {
            if ((typeof(val.enabled) === "undefined") || (val.enabled)) {
                var option = $(document.createElement('option'))
                                .attr('value', i)
                                .html(val.title);
                if (i === selectedvalue) {
                    option.attr('selected', 'selected');
                }
                if (typeof(val.group) !== "undefined") {
                    var optgroup = select.find("optgroup[label=" + val.group + "]");
                    if (optgroup.length === 0) {
                        optgroup = $(document.createElement('optgroup'))
                                    .attr('label', val.group);
                        optgroup.append(option);
                        select.append(optgroup);
                    } else {
                        optgroup.append(option);
                    }
                } else {
                    select.append(option);
                }
            }
        });
        return select;
    };

    // Create a queryindex select menu
    $.querywidget.createQueryIndex = function (value, fname) {
        return $.querywidget.createSelect($.querywidget.config.indexes,
                            value,
                            'queryindex',
                            fname + '.i:records');
    };

    // Create a queryoperator select menu
    $.querywidget.createQueryOperator = function (index, value, fname) {
        return $.querywidget.createSelect($.querywidget.config.indexes[index].operators,
                            value,
                            'queryoperator',
                            fname + '.o:records');
    };

    $.querywidget.createWidget = function (type, index, fname) {
        switch (type) {
            case 'StringWidget':
                return $(document.createElement('input'))
                    .attr({
                        'autocomplete': 'off',
                        'type': 'text',
                        'name': fname + '.v:records'
                    })
                    .addClass('querywidget queryvalue stringWidget');
            case 'DateWidget':
                return $(document.createElement('input'))
                    .attr({
                        'autocomplete': 'off',
                        'type': 'text',
                        'name': fname + '.v:records'
                    })
                    .addClass('querywidget queryvalue dateWidget date');
            case 'DateRangeWidget':
                return $(document.createElement('div'))
                    .addClass('querywidget dateRangeWidget')
                    .append($(document.createElement('input'))
                        .attr({
                            'autocomplete': 'off',
                            'type': 'text',
                            'name': fname + '.v:records:list'
                        })
                        .addClass('queryvalue date')
                    )
                    .append($(document.createElement('span'))
                        .html(' and ')
                    )
                    .append($(document.createElement('input'))
                        .attr({
                            'autocomplete': 'off',
                            'type': 'text',
                            'name': fname + '.v:records:list'
                        })
                        .addClass('queryvalue date')
                    );
            case 'RelativeDateWidget':
                return $(document.createElement('div'))
                    .addClass('querywidget relativeDateWidget')
                .append($(document.createElement('input'))
                        .attr({
                            'autocomplete': 'off',
                            'type': 'text',
                            'name': fname + '.v:records'
            })
            .addClass('queryvalue')
            )
                .append($(document.createElement('span'))
            .html('days')
            );
            case 'ReferenceWidget':
                return $(document.createElement('dl'))
                    .addClass('querywidget referenceWidget')
                    .append($(document.createElement('dt'))
                        .html('Select...')
                        .addClass('hiddenStructure')
                    )
                    .append($(document.createElement('dd'))
                        .append($(document.createElement('input'))
                            .attr({
                                'autocomplete': 'off',
                                'type': 'text',
                                'name': fname + '.v:records'
                            })
                            .addClass('queryvalue')
                        )
                    );
            case 'RelativePathWidget':
                return $(document.createElement('input'))
                    .attr({
                        'autocomplete': 'off',
                        'type': 'text',
                        'name': fname + '.v:records'
                    })
                    .addClass('querywidget queryvalue relativePathWidget');
            case 'MultipleSelectionWidget':
                var dl = $(document.createElement('dl'))
                    .addClass('querywidget multipleSelectionWidget')
                    .append($(document.createElement('dt'))
                        .append($(document.createElement('span'))
                            .addClass('arrowDownAlternative')
                            .html('&#09660;')
                        )
                        .append($(document.createElement('span'))
                            .html('Select...')
                            .addClass('multipleSelectionWidgetTitle')
                        )
                    );
                var dd = $(document.createElement('dd')).addClass('hiddenStructure widgetPulldownMenu');
                $.each($.querywidget.config.indexes[index].values, function (i, val) {
                    dd.append($(document.createElement('label'))
                        .append($(document.createElement('input'))
                            .attr({
                                'type': 'checkbox',
                                'name': fname + '.v:records:list',
                                'value': i
                            })
                        )
                        .append($(document.createElement('span'))
                            .html(val.title)
                        )
                    );
                });
                dl.append(dd);
                return dl;
            default:
                return $(document.createElement('div'))
                    .html('&nbsp;')
                    .addClass('querywidget queryvalue emptyWidget');
        }
    };

    $.querywidget.getCurrentWidget  = function (node) {
        var classes = node.attr('class').split(' ');
        var i;
        for(i = 0; i<classes.length; i++) {
            if (classes[i].indexOf('Widget') !== -1) {
                var classname = classes[i];
                return classname.slice(0,1).toUpperCase() + classname.slice(1);
            }
        }
    };

    $.querywidget.updateWidget = function (node) {
    if (typeof(node) === "undefined") {
        node = $('.querywidget');
    }
    if ($().dateinput) {
            $(node).parents('.criteria').find('.date').dateinput({change: function() { $.querywidget.updateSearch();}, firstDay: 1,selectors: true, trigger: false, yearRange: [-10, 10]}).unbind('change')
                .bind('onShow', function (event) {
                    var trigger_offset = $(this).next().offset();
                    $(this).data('dateinput').getCalendar().offset(
                        {top: trigger_offset.top+20, left: trigger_offset.left}
                    );
                });
        }
    };

    $.querywidget.updateSearch = function () {
        var context_url = (function() {
            var baseUrl, pieces, index;
            baseUrl = $('base').attr('href');
            if (!baseUrl) {
                pieces = window.location.href.split('/');
                pieces.pop();
                baseUrl = pieces.join('/');
            }
            index = baseUrl.lastIndexOf('/');
            if (index > -1 && baseUrl.lastIndexOf('edit') > index) {
                // The url is for an edit page, so we strip the last part,
                // otherwise we wrongly get /edit/@@querybuilder_html_results
                // 'edit' can be 'atct_edit' too.
                baseUrl = baseUrl.slice(0, index);
            }
            return baseUrl;
        })();
        var query = context_url + "/@@querybuilder_html_results?";
        var querylist  = [];
        var items = $('.QueryWidget .queryindex');
        if (!items.length) {
            return;
        }
        items.each(function () {
            var results = $(this).parents('.criteria').children('.queryresults');
            var index = $(this).val();
            var operator = $(this).parents('.criteria').children('.queryoperator').val();
            var widget = $.querywidget.config.indexes[index].operators[operator].widget;
            querylist.push('query.i:records=' + index);
            querylist.push('query.o:records=' + operator);
            switch (widget) {
                case 'DateRangeWidget':
                    var daterangewidget = $(this).parents('.criteria').find('.querywidget');
                    querylist.push('query.v:records:list=' + $(daterangewidget.children('input')[0]).val());
                    querylist.push('query.v:records:list=' + $(daterangewidget.children('input')[1]).val());
                    break;

                case 'MultipleSelectionWidget':
                    var multipleselectionwidget = $(this).parents('.criteria').find('.querywidget');
                    multipleselectionwidget.find('input:checked').each(function () {
                        querylist.push('query.v:records:list=' + $(this).val());
                    });
                    break;
                default:
                    querylist.push('query.v:records=' + $(this).parents('.criteria').find('.queryvalue').val());
                    break;
            }

            $.get(context_url + '/@@querybuildernumberofresults?' + querylist.join('&'),
                  {},
                  function (data) { results.html(data); });
        });
        query += querylist.join('&');
        query += '&sort_on=' + $('#sort_on').val();
        if ($('#sort_order:checked').length > 0) {
            query += '&sort_order=reverse';
        }
        $.get(query, {}, function (data) { $('.QueryWidget .previewresults').html(data); });
    };

    /* Clicking outside a multipleSelectionWidget will close all open
       multipleSelectionWidgets */
    $.querywidget.hideMultiSelectionWidgetEvent = function(event) {
        if ($(event.target).parents('.multipleSelectionWidget').length) {
            return;
        }
        $('.multipleSelectionWidget dd').addClass('hiddenStructure');
    }


    // Enhance for javascript browsers
    $(document).ready(function () {

        // Check if QueryWidget exists on page
        if ($(".QueryWidget").length === 0) {
            return false;
        }

        // Init
        $.querywidget.init();


        // We need two keep two fields for each sorting field ('#sort_on',
        // '#sort_reversed'). The query results preview that calls
        // '@@querybuilder_html_results' in plone.app.querystring expects a
        // sort_on and sort_order param. To store the actual setting on the
        // collection we need the two z3c.form-based fields
        // ('#form-widgets-ICollection-sort_on', '#form-widgets-ICollection-sort_reversed')

        // Synchronize the '#sort_on' field with the hidden
        // #form-widgets-ICollection-sort_on z3c.form field on load.
        $('#sort_on').val($('#form-widgets-ICollection-sort_on').val());

        // Synchronize the '#sort_order' field with the hidden
        // #form-widgets-ICollection-sort_reversed z3c.form field on load.
        if ($('#form-widgets-ICollection-sort_reversed-0').attr('checked')) {
            $('#sort_order').attr('checked', true);
        } else {
            $('#sort_order').attr('checked', false);
        }

        // Synchronize the z3c.form '#form-widgets-ICollection-sort_on' field
        // with the '#sort_on' field on user interaction
        $("div.QueryWidget").on('change', '#sort_on', function () {
            $('#form-widgets-ICollection-sort_on').val($(this).val());
        });

        // Synchronize the z3c.form '#form-widgets-ICollection-sort_reversed' field
        // with the '#sort_order' field on user interaction.
        $("div.QueryWidget").on('click', '#sort_order', function () {
            if ($(this).is(":checked")) {
                $('#form-widgets-ICollection-sort_reversed-0').attr('checked', true);
            } else {
                $('#form-widgets-ICollection-sort_reversed-0').attr('checked', false);
            }
        });

        // Hide the z3c.form widgets for sorting because they are only needed
        // internally.
        $('#formfield-form-widgets-ICollection-sort_on').hide();
        $('#formfield-form-widgets-ICollection-sort_reversed').hide();

    });

    // Init widget
    $.querywidget.init = function () {

        // Check if already initialized
        if ($.querywidget.initialized === true) {

            // Return nothing done
            return false;
        }

        // Set initialized
        $.querywidget.initialized = true;

        // Get configuration
        $.getJSON(portal_url + '/@@querybuilderjsonconfig', function (data) {
            $.querywidget.config = data;

            // Find querywidgets
            $(".QueryWidget").each(function () {

                // Get object
                var obj = $(this);
                var fname = obj.attr('data-fieldname');

                // Hide controls used for non-javascript only
                obj.find(".addIndexButton").hide();
                obj.find(".multipleSelectionWidget dt").removeClass('hiddenStructure');
                obj.find(".multipleSelectionWidget dd").addClass('hiddenStructure widgetPulldownMenu');

                $('div.queryindex').each(function () {
                    $(this).before(
                        $(document.createElement('div'))
                            .addClass('queryresults discreet')
                            .html('')
                    );
                    $(this).replaceWith($.querywidget.createQueryIndex($(this).children('input').val(), fname));
                });
                $('div.queryoperator').each(function () {
                    $(this).replaceWith($.querywidget.createQueryOperator($(this).parents('.criteria').children('.queryindex').val(),
                                                            $(this).children('input').val(), fname));
                });
                $.querywidget.updateSearch();
        $.querywidget.updateWidget();

            });
        });

        $("div.QueryWidget").on('click', '.multipleSelectionWidget dt', function () {
            var multiselectionwidget = $(this).parent().children('dd');
            if(!$(multiselectionwidget).hasClass('hiddenStructure')) {
                $(multiselectionwidget).addClass('hiddenStructure');
                $(window).unbind('click', $.querywidget.hideMultiSelectionWidgetEvent);
            } else {
                $(multiselectionwidget).removeClass('hiddenStructure');
                $(window).bind('click', $.querywidget.hideMultiSelectionWidgetEvent);
            }
        });

        $("div.QueryWidget").on('change', '.queryindex', function () {
            var fname = $(this).closest('.QueryWidget').attr('data-fieldname');
            var index = $(this).find(':selected')[0].value;
            $(this).parents(".criteria").children('.queryoperator')
                .replaceWith($.querywidget.createQueryOperator(index, '', fname));
            var operatorvalue = $(this).parents('.criteria').children('.queryoperator').val();
            var widget = $.querywidget.config.indexes[index].operators[operatorvalue].widget;
            var querywidget = $(this).parent(".criteria").children('.querywidget');
            if ((widget !== $.querywidget.getCurrentWidget(querywidget)) || (widget === 'MultipleSelectionWidget')) {
                querywidget.replaceWith($.querywidget.createWidget(widget, index, fname));
        $.querywidget.updateWidget($(this).parent(".criteria").children('.querywidget'));
            }
            $.querywidget.updateSearch();
        });

        $("div.QueryWidget").on('change', '.queryoperator', function () {
            var fname = $(this).closest('.QueryWidget').attr('data-fieldname');
            var index = $(this).parents('.criteria').children('.queryindex').val();
            var operatorvalue = $(this).children(':selected')[0].value;
            var widget = $.querywidget.config.indexes[index].operators[operatorvalue].widget;
            var querywidget = $(this).parent().children('.querywidget');
            if (widget !== $.querywidget.getCurrentWidget(querywidget)) {
                querywidget.replaceWith($.querywidget.createWidget(widget, index, fname));
        $.querywidget.updateWidget($(this).parent().children('.querywidget'));
            }
            $.querywidget.updateSearch();

        });

        $("div.QueryWidget").on('change', '#sort_on,#sort_order', function () {
            $.querywidget.updateSearch();
        });

        $("div.QueryWidget").on('change', '.multipleSelectionWidget input', function () {
            var widget = $(this).parents('.multipleSelectionWidget');
            var selected_values = [];
            widget.find('input:checked').each(function () {
                selected_values.push($(this).parent().children('span').html());
            });
            widget.find('.multipleSelectionWidgetTitle')
                .attr('title', selected_values.join(', '))
                .html(selected_values.join(', '));
            $.querywidget.updateSearch();
        });

        $("div.QueryWidget").on('keyup', '.queryvalue', function () {
            $.querywidget.updateSearch();
        });

        $("div.QueryWidget").on('keydown', '.queryvalue', function (e) {
            if (e.keyCode === 13) {
                return false;
            }
        });

        $("div.QueryWidget").on('change', '.addIndex', function () {
            var fname = $(this).closest('.QueryWidget').attr('data-fieldname');
            var index = $(this).find(':selected')[0].value;
            var criteria = $(this).parents('.criteria');
            var newcriteria = $(document.createElement('div'))
                                .addClass('criteria');

            newcriteria.append(
                    $(document.createElement('div'))
                        .addClass('queryresults discreet')
                        .html('')
                );
            newcriteria.append($.querywidget.createQueryIndex(index, fname));
            var operator = $.querywidget.createQueryOperator(index, '', fname);
            newcriteria.append(operator);
            var operatorvalue = $(operator.children()[0]).attr('value');
            newcriteria.append($.querywidget.createWidget($.querywidget.config.indexes[index].operators[operatorvalue].widget, index, fname));
            newcriteria.append(

                // How will we translate these values?

                $(document.createElement('input'))
                    .attr({
                        'value': 'Remove line',
                        'type': 'submit',
                        'name': 'removecriteria'
                    })
                    .addClass('removecriteria discreet')
            );
            criteria.before(newcriteria);
            $(this).val('');
            $.querywidget.updateSearch();
        });

        $("div.QueryWidget").on('click', '.removecriteria', function () {
            $(this).parents('.criteria').remove();
            $.querywidget.updateSearch();
            return false;
        });
    };

}(jQuery));


/* - ++resource++collective.galleria.js - */
// https://www.gov.br/portal_javascripts/++resource++collective.galleria.js?original=1
(function($){var undef,window=this,doc=window.document,$doc=$(doc),$win=$(window),protoArray=Array.prototype,VERSION=1.29,DEBUG=true,TIMEOUT=30000,DUMMY=false,NAV=navigator.userAgent.toLowerCase(),HASH=window.location.hash.replace(/#\//,''),F=function(){},FALSE=function(){return false},IE=(function(){var v=3,div=doc.createElement('div'),all=div.getElementsByTagName('i');do{div.innerHTML='<!--[if gt IE '+(++v)+']><i></i><![endif]-->'} while(all[0]);return v>4?v:undef}()),DOM=function(){return{html:doc.documentElement,body:doc.body,head:doc.getElementsByTagName('head')[0],title:doc.title}},IFRAME=window.parent!==window.self,_eventlist='data ready thumbnail loadstart loadfinish image play pause progress '+'fullscreen_enter fullscreen_exit idle_enter idle_exit rescale '+'lightbox_open lightbox_close lightbox_image',_events=(function(){var evs=[];$.each(_eventlist.split(' '), function(i,ev){evs.push(ev);if(/_/.test(ev)){evs.push(ev.replace(/_/g,''))}});return evs}()),_legacyOptions=function(options){var n;if(typeof options!=='object'){return options}
$.each(options, function(key,value){if(/^[a-z]+_/.test(key)){n='';$.each(key.split('_'), function(i,k){n+=i>0?k.substr(0,1).toUpperCase()+k.substr(1):k});options[n]=value;delete options[key]}});return options},_patchEvent=function(type){if($.inArray(type,_events)>-1){return Galleria[type.toUpperCase()]}
return type},_video={youtube:{reg:/https?:\/\/(?:[a-zA_Z]{2,3}.)?(?:youtube\.com\/watch\?)((?:[\w\d\-\_\=]+&amp;(?:amp;)?)*v(?:&lt;[A-Z]+&gt;)?=([0-9a-zA-Z\-\_]+))/i,embed: function(id){return 'http://www.youtube.com/embed/'+id},getThumb: function(id,success,fail){fail=fail||F;$.getJSON(window.location.protocol+'//gdata.youtube.com/feeds/api/videos/'+id+'?v=2&alt=json-in-script&callback=?', function(data){try{success(data.entry.media$group.media$thumbnail[0].url)} catch(e){fail()}}).error(fail)}},vimeo:{reg:/https?:\/\/(?:www\.)?(vimeo\.com)\/(?:hd#)?([0-9]+)/i,embed: function(id){return 'http://player.vimeo.com/video/'+id},getThumb: function(id,success,fail){fail=fail||F;$.getJSON('http://vimeo.com/api/v2/video/'+id+'.json?callback=?', function(data){try{success(data[0].thumbnail_medium)} catch(e){fail()}}).error(fail)}},dailymotion:{reg:/https?:\/\/(?:www\.)?(dailymotion\.com)\/video\/([^_]+)/,embed: function(id){return 'http://www.dailymotion.com/embed/video/'+id},getThumb: function(id,success,fail){fail=fail||F;$.getJSON('https://api.dailymotion.com/video/'+id+'?fields=thumbnail_medium_url&callback=?', function(data){try{success(data.thumbnail_medium_url)} catch(e){fail()}}).error(fail)}}},_videoTest=function(url){var match;for(var v in _video){match=url&&url.match(_video[v].reg);if(match&&match.length){return{id:match[2],provider:v}}}
return false},_nativeFullscreen={support:(function(){var html=DOM().html;return!IFRAME&&(html.requestFullscreen||html.mozRequestFullScreen||html.webkitRequestFullScreen)}()),callback:F,enter: function(instance,callback,elem){this.instance=instance;this.callback=callback||F;elem=elem||DOM().html;if(elem.requestFullscreen){elem.requestFullscreen()}
else if(elem.mozRequestFullScreen){elem.mozRequestFullScreen()}
else if(elem.webkitRequestFullScreen){elem.webkitRequestFullScreen()}},exit: function(callback){this.callback=callback||F;if(doc.exitFullscreen){doc.exitFullscreen()}
else if(doc.mozCancelFullScreen){doc.mozCancelFullScreen()}
else if(doc.webkitCancelFullScreen){doc.webkitCancelFullScreen()}},instance:null,listen: function(){if(!this.support){return}
var handler=function(){if(!_nativeFullscreen.instance){return}
var fs=_nativeFullscreen.instance._fullscreen;if(doc.fullscreen||doc.mozFullScreen||doc.webkitIsFullScreen){fs._enter(_nativeFullscreen.callback)} else{fs._exit(_nativeFullscreen.callback)}};doc.addEventListener('fullscreenchange',handler,false);doc.addEventListener('mozfullscreenchange',handler,false);doc.addEventListener('webkitfullscreenchange',handler,false)}},_galleries=[],_instances=[],_hasError=false,_canvas=false,_pool=[],_themeLoad=function(theme){Galleria.theme=theme;$.each(_pool, function(i,instance){if(!instance._initialized){instance._init.call(instance)}});_pool=[]},Utils=(function(){return{clearTimer: function(id){$.each(Galleria.get(), function(){this.clearTimer(id)})},addTimer: function(id){$.each(Galleria.get(), function(){this.addTimer(id)})},array: function(obj){return protoArray.slice.call(obj,0)},create: function(className,nodeName){nodeName=nodeName||'div';var elem=doc.createElement(nodeName);elem.className=className;return elem},removeFromArray: function(arr,elem){$.each(arr, function(i,el){if(el==elem){arr.splice(i,1);return false}});return arr},getScriptPath: function(src){src=src||$('script:last').attr('src');var slices=src.split('/');if(slices.length==1){return ''}
slices.pop();return slices.join('/')+'/'},animate:(function(){var transition=(function(style){var props='transition WebkitTransition MozTransition OTransition'.split(' '),i;if(window.opera){return false}
for(i=0;props[i];i++){if(typeof style[props[i]]!=='undefined'){return props[i]}}
return false}((doc.body||doc.documentElement).style));var endEvent={MozTransition:'transitionend',OTransition:'oTransitionEnd',WebkitTransition:'webkitTransitionEnd',transition:'transitionend'}[transition];var easings={_default:[0.25,0.1,0.25,1],galleria:[0.645,0.045,0.355,1],galleriaIn:[0.55,0.085,0.68,0.53],galleriaOut:[0.25,0.46,0.45,0.94],ease:[0.25,0,0.25,1],linear:[0.25,0.25,0.75,0.75],'ease-in':[0.42,0,1,1],'ease-out':[0,0,0.58,1],'ease-in-out':[0.42,0,0.58,1]};var setStyle=function(elem,value,suffix){var css={};suffix=suffix||'transition';$.each('webkit moz ms o'.split(' '), function(){css['-'+this+'-'+suffix]=value});elem.css(css)};var clearStyle=function(elem){setStyle(elem,'none','transition');if(Galleria.WEBKIT&&Galleria.TOUCH){setStyle(elem,'translate3d(0,0,0)','transform');if(elem.data('revert')){elem.css(elem.data('revert'));elem.data('revert',null)}}};var change,strings,easing,syntax,revert,form,css;return function(elem,to,options){options=$.extend({duration:400,complete:F,stop:false},options);elem=$(elem);if(!options.duration){elem.css(to);options.complete.call(elem[0]);return}
if(!transition){elem.animate(to,options);return}
if(options.stop){elem.unbind(endEvent);clearStyle(elem)}
change=false;$.each(to, function(key,val){css=elem.css(key);if(Utils.parseValue(css)!=Utils.parseValue(val)){change=true}
elem.css(key,css)});if(!change){window.setTimeout( function(){options.complete.call(elem[0])},options.duration);return}
strings=[];easing=options.easing in easings?easings[options.easing]:easings._default;syntax=' '+options.duration+'ms'+' cubic-bezier('+easing.join(',')+')';window.setTimeout((function(elem,endEvent,to,syntax){return function(){elem.one(endEvent,(function(elem){return function(){clearStyle(elem);options.complete.call(elem[0])}}(elem)));if(Galleria.WEBKIT&&Galleria.TOUCH){revert={};form=[0,0,0];$.each(['left','top'], function(i,m){if(m in to){form[i]=(Utils.parseValue(to[m])-Utils.parseValue(elem.css(m)))+'px';revert[m]=to[m];delete to[m]}});if(form[0]||form[1]){elem.data('revert',revert);strings.push('-webkit-transform'+syntax);setStyle(elem,'translate3d('+form.join(',')+')','transform')}}
$.each(to, function(p,val){strings.push(p+syntax)});setStyle(elem,strings.join(','));elem.css(to)}}(elem,endEvent,to,syntax)),2)}}()),removeAlpha: function(elem){if(IE<9&&elem){var style=elem.style,currentStyle=elem.currentStyle,filter=currentStyle&&currentStyle.filter||style.filter||"";if(/alpha/.test(filter)){style.filter=filter.replace(/alpha\([^)]*\)/i,'')}}},forceStyles: function(elem,styles){elem=$(elem);if(elem.attr('style')){elem.data('styles',elem.attr('style')).removeAttr('style')}
elem.css(styles)},revertStyles: function(){$.each(Utils.array(arguments), function(i,elem){elem=$(elem);elem.removeAttr('style');elem.attr('style','');if(elem.data('styles')){elem.attr('style',elem.data('styles')).data('styles',null)}})},moveOut: function(elem){Utils.forceStyles(elem,{position:'absolute',left:-10000})},moveIn: function(){Utils.revertStyles.apply(Utils,Utils.array(arguments))},elem: function(elem){if(elem instanceof $){return{$:elem,dom:elem[0]}} else{return{$:$(elem),dom:elem}}},hide: function(elem,speed,callback){callback=callback||F;var el=Utils.elem(elem),$elem=el.$;elem=el.dom;if(!$elem.data('opacity')){$elem.data('opacity',$elem.css('opacity'))}
var style={opacity:0};if(speed){var complete=IE<9&&elem? function(){Utils.removeAlpha(elem);elem.style.visibility='hidden';callback.call(elem)}:callback;Utils.animate(elem,style,{duration:speed,complete:complete,stop:true})} else{if(IE<9&&elem){Utils.removeAlpha(elem);elem.style.visibility='hidden'} else{$elem.css(style)}}},show: function(elem,speed,callback){callback=callback||F;var el=Utils.elem(elem),$elem=el.$;elem=el.dom;var saved=parseFloat($elem.data('opacity'))||1,style={opacity:saved};if(speed){if(IE<9){$elem.css('opacity',0);elem.style.visibility='visible'}
var complete=IE<9&&elem? function(){if(style.opacity==1){Utils.removeAlpha(elem)}
callback.call(elem)}:callback;Utils.animate(elem,style,{duration:speed,complete:complete,stop:true})} else{if(IE<9&&style.opacity==1&&elem){Utils.removeAlpha(elem);elem.style.visibility='visible'} else{$elem.css(style)}}},optimizeTouch:(function(){var node,evs,fakes,travel,evt={},handler=function(e){e.preventDefault();evt=$.extend({},e,true)},attach=function(){this.evt=evt},fake=function(){this.handler.call(node,this.evt)};return function(elem){$(elem).bind('touchend', function(e){node=e.target;travel=true;while(node.parentNode&&node!=e.currentTarget&&travel){evs=$(node).data('events');fakes=$(node).data('fakes');if(evs&&'click' in evs){travel=false;e.preventDefault();$(node).click(handler).click();evs.click.pop();$.each(evs.click,attach);$(node).data('fakes',evs.click);delete evs.click} else if(fakes){travel=false;e.preventDefault();$.each(fakes,fake)}
node=node.parentNode}})}}()),wait: function(options){options=$.extend({until:FALSE,success:F,error: function(){Galleria.raise('Could not complete wait function.')},timeout:3000},options);var start=Utils.timestamp(),elapsed,now,fn=function(){now=Utils.timestamp();elapsed=now-start;if(options.until(elapsed)){options.success();return false}
if(typeof options.timeout=='number'&&now>=start+options.timeout){options.error();return false}
window.setTimeout(fn,10)};window.setTimeout(fn,10)},toggleQuality: function(img,force){if((IE!==7&&IE!==8)||!img||img.nodeName.toUpperCase()!='IMG'){return}
if(typeof force==='undefined'){force=img.style.msInterpolationMode==='nearest-neighbor'}
img.style.msInterpolationMode=force?'bicubic':'nearest-neighbor'},insertStyleTag: function(styles,id){if(id&&$('#'+id).length){return}
var style=doc.createElement('style');if(id){style.id=id}
DOM().head.appendChild(style);if(style.styleSheet){style.styleSheet.cssText=styles} else{var cssText=doc.createTextNode(styles);style.appendChild(cssText)}},loadScript: function(url,callback){var done=false,script=$('<scr'+'ipt>').attr({src:url,async:true}).get(0);script.onload=script.onreadystatechange=function(){if(!done&&(!this.readyState||this.readyState==='loaded'||this.readyState==='complete')){done=true;script.onload=script.onreadystatechange=null;if(typeof callback==='function'){callback.call(this,this)}}};DOM().head.appendChild(script)},parseValue: function(val){if(typeof val==='number'){return val} else if(typeof val==='string'){var arr=val.match(/\-?\d|\./g);return arr&&arr.constructor===Array?arr.join('')*1:0} else{return 0}},timestamp: function(){return new Date().getTime()},loadCSS: function(href,id,callback){var link,length;$('link[rel=stylesheet]').each(function(){if(new RegExp(href).test(this.href)){link=this;return false}});if(typeof id==='function'){callback=id;id=undef}
callback=callback||F;if(link){callback.call(link,link);return link}
length=doc.styleSheets.length;if($('#'+id).length){$('#'+id).attr('href',href);length--} else{link=$('<link>').attr({rel:'stylesheet',href:href,id:id}).get(0);var styles=$('link[rel="stylesheet"], style');if(styles.length){styles.get(0).parentNode.insertBefore(link,styles[0])} else{DOM().head.appendChild(link)}
if(IE){if(length>=31){Galleria.raise('You have reached the browser stylesheet limit (31)',true);return}}}
if(typeof callback==='function'){var $loader=$('<s>').attr('id','galleria-loader').hide().appendTo(DOM().body);Utils.wait({until: function(){return $loader.height()==1},success: function(){$loader.remove();callback.call(link,link)},error: function(){$loader.remove();Galleria.raise('Theme CSS could not load after 20 sec. '+Galleria.QUIRK?'Your browser is in Quirks Mode, please add a correct doctype.':'Please download the latest theme at http://galleria.io/customer/.',true)},timeout:20000})}
return link}}}()),_transitions=(function(){var _slide=function(params,complete,fade,door){var easing=this.getOptions('easing'),distance=this.getStageWidth(),from={left:distance *(params.rewind?-1:1)},to={left:0};if(fade){from.opacity=0;to.opacity=1} else{from.opacity=1}
$(params.next).css(from);Utils.animate(params.next,to,{duration:params.speed,complete:(function(elems){return function(){complete();elems.css({left:0})}}($(params.next).add(params.prev))),queue:false,easing:easing});if(door){params.rewind=!params.rewind}
if(params.prev){from={left:0};to={left:distance *(params.rewind?1:-1)};if(fade){from.opacity=1;to.opacity=0}
$(params.prev).css(from);Utils.animate(params.prev,to,{duration:params.speed,queue:false,easing:easing,complete: function(){$(this).css('opacity',0)}})}};return{active:false,init: function(effect,params,complete){if(_transitions.effects.hasOwnProperty(effect)){_transitions.effects[effect].call(this,params,complete)}},effects:{fade: function(params,complete){$(params.next).css({opacity:0,left:0});Utils.animate(params.next,{opacity:1},{duration:params.speed,complete:complete});if(params.prev){$(params.prev).css('opacity',1).show();Utils.animate(params.prev,{opacity:0},{duration:params.speed})}},flash: function(params,complete){$(params.next).css({opacity:0,left:0});if(params.prev){Utils.animate(params.prev,{opacity:0},{duration:params.speed/2,complete: function(){Utils.animate(params.next,{opacity:1},{duration:params.speed,complete:complete})}})} else{Utils.animate(params.next,{opacity:1},{duration:params.speed,complete:complete})}},pulse: function(params,complete){if(params.prev){$(params.prev).hide()}
$(params.next).css({opacity:0,left:0}).show();Utils.animate(params.next,{opacity:1},{duration:params.speed,complete:complete})},slide: function(params,complete){_slide.apply(this,Utils.array(arguments))},fadeslide: function(params,complete){_slide.apply(this,Utils.array(arguments).concat([true]))},doorslide: function(params,complete){_slide.apply(this,Utils.array(arguments).concat([false,true]))}}}}());_nativeFullscreen.listen();Galleria=function(){var self=this;this._options={};this._playing=false;this._playtime=5000;this._active=null;this._queue={length:0};this._data=[];this._dom={};this._thumbnails=[];this._layers=[];this._initialized=false;this._firstrun=false;this._stageWidth=0;this._stageHeight=0;this._target=undef;this._binds=[];this._id=parseInt(Math.random()*10000,10);var divs='container stage images image-nav image-nav-left image-nav-right '+'info info-text info-title info-description '+'thumbnails thumbnails-list thumbnails-container thumb-nav-left thumb-nav-right '+'loader counter tooltip',spans='current total';$.each(divs.split(' '), function(i,elemId){self._dom[elemId]=Utils.create('galleria-'+elemId)});$.each(spans.split(' '), function(i,elemId){self._dom[elemId]=Utils.create('galleria-'+elemId,'span')});var keyboard=this._keyboard={keys:{'UP':38,'DOWN':40,'LEFT':37,'RIGHT':39,'RETURN':13,'ESCAPE':27,'BACKSPACE':8,'SPACE':32},map:{},bound:false,press: function(e){var key=e.keyCode||e.which;if(key in keyboard.map&&typeof keyboard.map[key]==='function'){keyboard.map[key].call(self,e)}},attach: function(map){var key,up;for(key in map){if(map.hasOwnProperty(key)){up=key.toUpperCase();if(up in keyboard.keys){keyboard.map[keyboard.keys[up]]=map[key]} else{keyboard.map[up]=map[key]}}}
if(!keyboard.bound){keyboard.bound=true;$doc.bind('keydown',keyboard.press)}},detach: function(){keyboard.bound=false;keyboard.map={};$doc.unbind('keydown',keyboard.press)}};var controls=this._controls={0:undef,1:undef,active:0,swap: function(){controls.active=controls.active?0:1},getActive: function(){return controls[controls.active]},getNext: function(){return controls[1-controls.active]}};var carousel=this._carousel={next:self.$('thumb-nav-right'),prev:self.$('thumb-nav-left'),width:0,current:0,max:0,hooks:[],update: function(){var w=0,h=0,hooks=[0];$.each(self._thumbnails, function(i,thumb){if(thumb.ready){w+=thumb.outerWidth||$(thumb.container).outerWidth(true);hooks[i+1]=w;h=Math.max(h,thumb.outerHeight||$(thumb.container).outerHeight(true))}});self.$('thumbnails').css({width:w,height:h});carousel.max=w;carousel.hooks=hooks;carousel.width=self.$('thumbnails-list').width();carousel.setClasses();self.$('thumbnails-container').toggleClass('galleria-carousel',w>carousel.width);carousel.width=self.$('thumbnails-list').width()},bindControls: function(){var i;carousel.next.bind('click', function(e){e.preventDefault();if(self._options.carouselSteps==='auto'){for(i=carousel.current;i<carousel.hooks.length;i++){if(carousel.hooks[i]-carousel.hooks[carousel.current]>carousel.width){carousel.set(i-2);break}}} else{carousel.set(carousel.current+self._options.carouselSteps)}});carousel.prev.bind('click', function(e){e.preventDefault();if(self._options.carouselSteps==='auto'){for(i=carousel.current;i>=0;i--){if(carousel.hooks[carousel.current]-carousel.hooks[i]>carousel.width){carousel.set(i+2);break} else if(i===0){carousel.set(0);break}}} else{carousel.set(carousel.current-self._options.carouselSteps)}})},set: function(i){i=Math.max(i,0);while(carousel.hooks[i-1]+carousel.width>=carousel.max&&i>=0){i--}
carousel.current=i;carousel.animate()},getLast: function(i){return(i||carousel.current)-1},follow: function(i){if(i===0||i===carousel.hooks.length-2){carousel.set(i);return}
var last=carousel.current;while(carousel.hooks[last]-carousel.hooks[carousel.current]<carousel.width&&last<=carousel.hooks.length){last++}
if(i-1<carousel.current){carousel.set(i-1)} else if(i+2>last){carousel.set(i-last+carousel.current+2)}},setClasses: function(){carousel.prev.toggleClass('disabled',!carousel.current);carousel.next.toggleClass('disabled',carousel.hooks[carousel.current]+carousel.width>=carousel.max)},animate: function(to){carousel.setClasses();var num=carousel.hooks[carousel.current] *-1;if(isNaN(num)){return}
Utils.animate(self.get('thumbnails'),{left:num},{duration:self._options.carouselSpeed,easing:self._options.easing,queue:false})}};var tooltip=this._tooltip={initialized:false,open:false,timer:'tooltip'+self._id,swapTimer:'swap'+self._id,init: function(){tooltip.initialized=true;var css='.galleria-tooltip{padding:3px 8px;max-width:50%;background:#ffe;color:#000;z-index:3;position:absolute;font-size:11px;line-height:1.3;'+'opacity:0;box-shadow:0 0 2px rgba(0,0,0,.4);-moz-box-shadow:0 0 2px rgba(0,0,0,.4);-webkit-box-shadow:0 0 2px rgba(0,0,0,.4);}';Utils.insertStyleTag(css,'galleria-tooltip');self.$('tooltip').css({opacity:0.8,visibility:'visible',display:'none'})},move: function(e){var mouseX=self.getMousePosition(e).x,mouseY=self.getMousePosition(e).y,$elem=self.$('tooltip'),x=mouseX,y=mouseY,height=$elem.outerHeight(true)+1,width=$elem.outerWidth(true),limitY=height+15;var maxX=self.$('container').width()-width-2,maxY=self.$('container').height()-height-2;if(!isNaN(x)&&!isNaN(y)){x+=10;y-=(height+8);x=Math.max(0,Math.min(maxX,x));y=Math.max(0,Math.min(maxY,y));if(mouseY<limitY){y=limitY}
$elem.css({left:x,top:y})}},bind: function(elem,value){if(Galleria.TOUCH){return}
if(!tooltip.initialized){tooltip.init()}
var mouseout=function(){self.$('container').unbind('mousemove',tooltip.move);self.clearTimer(tooltip.timer);self.$('tooltip').stop().animate({opacity:0},200, function(){self.$('tooltip').hide();self.addTimer(tooltip.swapTimer, function(){tooltip.open=false},1000)})};var hover=function(elem,value){tooltip.define(elem,value);$(elem).hover(function(){self.clearTimer(tooltip.swapTimer);self.$('container').unbind('mousemove',tooltip.move).bind('mousemove',tooltip.move).trigger('mousemove');tooltip.show(elem);self.addTimer(tooltip.timer, function(){self.$('tooltip').stop().show().animate({opacity:1});tooltip.open=true},tooltip.open?0:500)},mouseout).click(mouseout)};if(typeof value==='string'){hover((elem in self._dom?self.get(elem):elem),value)} else{$.each(elem, function(elemID,val){hover(self.get(elemID),val)})}},show: function(elem){elem=$(elem in self._dom?self.get(elem):elem);var text=elem.data('tt'),mouseup=function(e){window.setTimeout((function(ev){return function(){tooltip.move(ev)}}(e)),10);elem.unbind('mouseup',mouseup)};text=typeof text==='function'?text():text;if(!text){return}
self.$('tooltip').html(text.replace(/\s/,'&#160;'));elem.bind('mouseup',mouseup)},define: function(elem,value){if(typeof value!=='function'){var s=value;value=function(){return s}}
elem=$(elem in self._dom?self.get(elem):elem).data('tt',value);tooltip.show(elem)}};var fullscreen=this._fullscreen={scrolled:0,crop:undef,active:false,keymap:self._keyboard.map,parseCallback: function(callback,enter){return _transitions.active? function(){if(typeof callback=='function'){callback.call(self)}
var active=self._controls.getActive(),next=self._controls.getNext();self._scaleImage(next);self._scaleImage(active);if(enter&&self._options.trueFullscreen){$(active.container).add(next.container).trigger('transitionend')}}:callback},enter: function(callback){callback=fullscreen.parseCallback(callback,true);if(self._options.trueFullscreen&&_nativeFullscreen.support){fullscreen.active=true;Utils.forceStyles(self.get('container'),{width:'100%',height:'100%'});self.rescale();if(Galleria.MAC){if(Galleria.WEBKIT){self.$('container').css('opacity',0).addClass('fullscreen');window.setTimeout(function(){fullscreen.scale();self.$('container').css('opacity',1)},50)} else{self.$('stage').css('opacity',0);window.setTimeout(function(){fullscreen.scale();self.$('stage').css('opacity',1)},4)}} else{self.$('container').addClass('fullscreen')}
$win.resize(fullscreen.scale);_nativeFullscreen.enter(self,callback,self.get('container'))} else{fullscreen.scrolled=$win.scrollTop();window.scrollTo(0,0);fullscreen._enter(callback)}},_enter: function(callback){fullscreen.active=true;if(IFRAME){fullscreen.iframe=(function(){var elem,refer=doc.referrer,test=doc.createElement('a'),loc=window.location;test.href=refer;if(test.protocol!=loc.protocol||test.hostname!=loc.hostname||test.port!=loc.port){Galleria.raise('Parent fullscreen not available. Iframe protocol, domains and ports must match.');return false}
fullscreen.pd=window.parent.document;$(fullscreen.pd).find('iframe').each(function(){var idoc=this.contentDocument||this.contentWindow.document;if(idoc===doc){elem=this;return false}});return elem}())}
Utils.hide(self.getActiveImage());if(IFRAME&&fullscreen.iframe){fullscreen.iframe.scrolled=$(window.parent).scrollTop();window.parent.scrollTo(0,0)}
var data=self.getData(),options=self._options,inBrowser=!self._options.trueFullscreen||!_nativeFullscreen.support,htmlbody={height:'100%',overflow:'hidden',margin:0,padding:0};if(inBrowser){self.$('container').addClass('fullscreen');Utils.forceStyles(self.get('container'),{position:'fixed',top:0,left:0,width:'100%',height:'100%',zIndex:10000});Utils.forceStyles(DOM().html,htmlbody);Utils.forceStyles(DOM().body,htmlbody)}
if(IFRAME&&fullscreen.iframe){Utils.forceStyles(fullscreen.pd.documentElement,htmlbody);Utils.forceStyles(fullscreen.pd.body,htmlbody);Utils.forceStyles(fullscreen.iframe,$.extend(htmlbody,{width:'100%',height:'100%',top:0,left:0,position:'fixed',zIndex:10000,border:'none'}))}
fullscreen.keymap=$.extend({},self._keyboard.map);self.attachKeyboard({escape:self.exitFullscreen,right:self.next,left:self.prev});fullscreen.crop=options.imageCrop;if(options.fullscreenCrop!=undef){options.imageCrop=options.fullscreenCrop}
if(data&&data.big&&data.image!==data.big){var big=new Galleria.Picture(),cached=big.isCached(data.big),index=self.getIndex(),thumb=self._thumbnails[index];self.trigger({type:Galleria.LOADSTART,cached:cached,rewind:false,index:index,imageTarget:self.getActiveImage(),thumbTarget:thumb,galleriaData:data});big.load(data.big, function(big){self._scaleImage(big,{complete: function(big){self.trigger({type:Galleria.LOADFINISH,cached:cached,index:index,rewind:false,imageTarget:big.image,thumbTarget:thumb});var image=self._controls.getActive().image;if(image){$(image).width(big.image.width).height(big.image.height).attr('style',$(big.image).attr('style')).attr('src',big.image.src)}}})})}
self.rescale(function(){self.addTimer(false, function(){if(inBrowser){Utils.show(self.getActiveImage())}
if(typeof callback==='function'){callback.call(self)}},100);self.trigger(Galleria.FULLSCREEN_ENTER)});if(!inBrowser){Utils.show(self.getActiveImage())} else{$win.resize(fullscreen.scale)}},scale: function(){self.rescale()},exit: function(callback){callback=fullscreen.parseCallback(callback);if(self._options.trueFullscreen&&_nativeFullscreen.support){_nativeFullscreen.exit(callback)} else{fullscreen._exit(callback)}},_exit: function(callback){fullscreen.active=false;var inBrowser=!self._options.trueFullscreen||!_nativeFullscreen.support;self.$('container').removeClass('fullscreen');if(inBrowser){Utils.hide(self.getActiveImage());Utils.revertStyles(self.get('container'),DOM().html,DOM().body);window.scrollTo(0,fullscreen.scrolled)}
if(IFRAME&&fullscreen.iframe){Utils.revertStyles(fullscreen.pd.documentElement,fullscreen.pd.body,fullscreen.iframe);if(fullscreen.iframe.scrolled){window.parent.scrollTo(0,fullscreen.iframe.scrolled)}}
self.detachKeyboard();self.attachKeyboard(fullscreen.keymap);self._options.imageCrop=fullscreen.crop;var big=self.getData().big,image=self._controls.getActive().image;if(!self.getData().iframe&&image&&big&&big==image.src){window.setTimeout(function(src){return function(){image.src=src}}(self.getData().image),1)}
self.rescale(function(){self.addTimer(false, function(){if(inBrowser){Utils.show(self.getActiveImage())}
if(typeof callback==='function'){callback.call(self)}
$win.trigger('resize')},50);self.trigger(Galleria.FULLSCREEN_EXIT)});$win.unbind('resize',fullscreen.scale)}};var idle=this._idle={trunk:[],bound:false,active:false,add: function(elem,to,from,hide){if(!elem){return}
if(!idle.bound){idle.addEvent()}
elem=$(elem);if(typeof from=='boolean'){hide=from;from={}}
from=from||{};var extract={},style;for(style in to){if(to.hasOwnProperty(style)){extract[style]=elem.css(style)}}
elem.data('idle',{from:$.extend(extract,from),to:to,complete:true,busy:false});if(!hide){idle.addTimer()} else{elem.css(to)}
idle.trunk.push(elem)},remove: function(elem){elem=$(elem);$.each(idle.trunk, function(i,el){if(el&&el.length&&!el.not(elem).length){elem.css(elem.data('idle').from);idle.trunk.splice(i,1)}});if(!idle.trunk.length){idle.removeEvent();self.clearTimer(idle.timer)}},addEvent: function(){idle.bound=true;self.$('container').bind('mousemove click',idle.showAll);if(self._options.idleMode=='hover'){self.$('container').bind('mouseleave',idle.hide)}},removeEvent: function(){idle.bound=false;self.$('container').bind('mousemove click',idle.showAll);if(self._options.idleMode=='hover'){self.$('container').unbind('mouseleave',idle.hide)}},addTimer: function(){if(self._options.idleMode=='hover'){return}
self.addTimer('idle', function(){idle.hide()},self._options.idleTime)},hide: function(){if(!self._options.idleMode||self.getIndex()===false||self.getData().iframe){return}
self.trigger(Galleria.IDLE_ENTER);var len=idle.trunk.length;$.each(idle.trunk, function(i,elem){var data=elem.data('idle');if(!data){return}
elem.data('idle').complete=false;Utils.animate(elem,data.to,{duration:self._options.idleSpeed,complete: function(){if(i==len-1){idle.active=false}}})})},showAll: function(){self.clearTimer('idle');$.each(idle.trunk, function(i,elem){idle.show(elem)})},show: function(elem){var data=elem.data('idle');if(!idle.active||(!data.busy&&!data.complete)){data.busy=true;self.trigger(Galleria.IDLE_EXIT);self.clearTimer('idle');Utils.animate(elem,data.from,{duration:self._options.idleSpeed/2,complete: function(){idle.active=true;$(elem).data('idle').busy=false;$(elem).data('idle').complete=true}})}
idle.addTimer()}};var lightbox=this._lightbox={width:0,height:0,initialized:false,active:null,image:null,elems:{},keymap:false,init: function(){self.trigger(Galleria.LIGHTBOX_OPEN);if(lightbox.initialized){return}
lightbox.initialized=true;var elems='overlay box content shadow title info close prevholder prev nextholder next counter image',el={},op=self._options,css='',abs='position:absolute;',prefix='lightbox-',cssMap={overlay:'position:fixed;display:none;opacity:'+op.overlayOpacity+';filter:alpha(opacity='+(op.overlayOpacity*100)+');top:0;left:0;width:100%;height:100%;background:'+op.overlayBackground+';z-index:99990',box:'position:fixed;display:none;width:400px;height:400px;top:50%;left:50%;margin-top:-200px;margin-left:-200px;z-index:99991',shadow:abs+'background:#000;width:100%;height:100%;',content:abs+'background-color:#fff;top:10px;left:10px;right:10px;bottom:10px;overflow:hidden',info:abs+'bottom:10px;left:10px;right:10px;color:#444;font:11px/13px arial,sans-serif;height:13px',close:abs+'top:10px;right:10px;height:20px;width:20px;background:#fff;text-align:center;cursor:pointer;color:#444;font:16px/22px arial,sans-serif;z-index:99999',image:abs+'top:10px;left:10px;right:10px;bottom:30px;overflow:hidden;display:block;',prevholder:abs+'width:50%;top:0;bottom:40px;cursor:pointer;',nextholder:abs+'width:50%;top:0;bottom:40px;right:-1px;cursor:pointer;',prev:abs+'top:50%;margin-top:-20px;height:40px;width:30px;background:#fff;left:20px;display:none;text-align:center;color:#000;font:bold 16px/36px arial,sans-serif',next:abs+'top:50%;margin-top:-20px;height:40px;width:30px;background:#fff;right:20px;left:auto;display:none;font:bold 16px/36px arial,sans-serif;text-align:center;color:#000',title:'float:left',counter:'float:right;margin-left:8px;'},hover=function(elem){return elem.hover(
function(){$(this).css('color','#bbb')},
function(){$(this).css('color','#444')})},appends={};if(IE&&IE>7){cssMap.nextholder+='background:#000;filter:alpha(opacity=0);';cssMap.prevholder+='background:#000;filter:alpha(opacity=0);'}
$.each(cssMap, function(key,value){css+='.galleria-'+prefix+key+'{'+value+'}'});css+='.galleria-'+prefix+'box.iframe .galleria-'+prefix+'prevholder,'+'.galleria-'+prefix+'box.iframe .galleria-'+prefix+'nextholder{'+'width:100px;height:100px;top:50%;margin-top:-70px}';Utils.insertStyleTag(css,'galleria-lightbox');$.each(elems.split(' '), function(i,elemId){self.addElement('lightbox-'+elemId);el[elemId]=lightbox.elems[elemId]=self.get('lightbox-'+elemId)});lightbox.image=new Galleria.Picture();$.each({box:'shadow content close prevholder nextholder',info:'title counter',content:'info image',prevholder:'prev',nextholder:'next'}, function(key,val){var arr=[];$.each(val.split(' '), function(i,prop){arr.push(prefix+prop)});appends[prefix+key]=arr});self.append(appends);$(el.image).append(lightbox.image.container);$(DOM().body).append(el.overlay,el.box);Utils.optimizeTouch(el.box);hover($(el.close).bind('click',lightbox.hide).html('&#215;'));$.each(['Prev','Next'], function(i,dir){var $d=$(el[dir.toLowerCase()]).html(/v/.test(dir)?'&#8249;&#160;':'&#160;&#8250;'),$e=$(el[dir.toLowerCase()+'holder']);$e.bind('click', function(){lightbox['show'+dir]()});if(IE<8||Galleria.TOUCH){$d.show();return}
$e.hover( function(){$d.show()}, function(e){$d.stop().fadeOut(200)})});$(el.overlay).bind('click',lightbox.hide);if(Galleria.IPAD){self._options.lightboxTransitionSpeed=0}},rescale: function(event){var width=Math.min($win.width()-40,lightbox.width),height=Math.min($win.height()-60,lightbox.height),ratio=Math.min(width/ lightbox.width, height / lightbox.height),destWidth=Math.round(lightbox.width * ratio)+40,destHeight=Math.round(lightbox.height * ratio)+60,to={width:destWidth,height:destHeight,'margin-top':Math.ceil(destHeight/2) *-1,'margin-left':Math.ceil(destWidth/2) *-1};if(event){$(lightbox.elems.box).css(to)} else{$(lightbox.elems.box).animate(to,{duration:self._options.lightboxTransitionSpeed,easing:self._options.easing,complete: function(){var image=lightbox.image,speed=self._options.lightboxFadeSpeed;self.trigger({type:Galleria.LIGHTBOX_IMAGE,imageTarget:image.image});$(image.container).show();$(image.image).animate({opacity:1},speed);Utils.show(lightbox.elems.info,speed)}})}},hide: function(){lightbox.image.image=null;$win.unbind('resize',lightbox.rescale);$(lightbox.elems.box).hide();Utils.hide(lightbox.elems.info);self.detachKeyboard();self.attachKeyboard(lightbox.keymap);lightbox.keymap=false;Utils.hide(lightbox.elems.overlay,200, function(){$(this).hide().css('opacity',self._options.overlayOpacity);self.trigger(Galleria.LIGHTBOX_CLOSE)})},showNext: function(){lightbox.show(self.getNext(lightbox.active))},showPrev: function(){lightbox.show(self.getPrev(lightbox.active))},show: function(index){lightbox.active=index=typeof index==='number'?index:self.getIndex()||0;if(!lightbox.initialized){lightbox.init()}
if(!lightbox.keymap){lightbox.keymap=$.extend({},self._keyboard.map);self.attachKeyboard({escape:lightbox.hide,right:lightbox.showNext,left:lightbox.showPrev})}
$win.unbind('resize',lightbox.rescale);var data=self.getData(index),total=self.getDataLength(),n=self.getNext(index),ndata,p,i;Utils.hide(lightbox.elems.info);try{for(i=self._options.preload;i>0;i--){p=new Galleria.Picture();ndata=self.getData(n);p.preload('big' in ndata?ndata.big:ndata.image);n=self.getNext(n)}} catch(e){}
lightbox.image.isIframe=!!data.iframe;$(lightbox.elems.box).toggleClass('iframe',!!data.iframe);lightbox.image.load(data.iframe||data.big||data.image, function(image){if(image.isIframe){var cw=$(window).width(),ch=$(window).height();if(self._options.maxVideoSize){var r=Math.min(self._options.maxVideoSize/cw,self._options.maxVideoSize/ch);if(r<1){cw *=r;ch *=r}}
lightbox.width=cw;lightbox.height=ch} else{lightbox.width=image.original.width;lightbox.height=image.original.height}
$(image.image).css({width:image.isIframe?'100%':'100.1%',height:image.isIframe?'100%':'100.1%',top:0,zIndex:99998,opacity:0,visibility:'visible'});lightbox.elems.title.innerHTML=data.title||'';lightbox.elems.counter.innerHTML=(index+1)+' / '+total;$win.resize(lightbox.rescale);lightbox.rescale()});$(lightbox.elems.overlay).show().css('visibility','visible');$(lightbox.elems.box).show()}};var _timer=this._timer={trunk:{},add: function(id,fn,delay,loop){id=id||new Date().getTime();loop=loop||false;this.clear(id);if(loop){var old=fn;fn=function(){old();_timer.add(id,fn,delay)}}
this.trunk[id]=window.setTimeout(fn,delay)},clear: function(id){var del=function(i){window.clearTimeout(this.trunk[i]);delete this.trunk[i]},i;if(!!id&&id in this.trunk){del.call(this,id)} else if(typeof id==='undefined'){for(i in this.trunk){if(this.trunk.hasOwnProperty(i)){del.call(this,i)}}}}};return this};Galleria.prototype={constructor:Galleria,init: function(target,options){var self=this;options=_legacyOptions(options);this._original={target:target,options:options,data:null};this._target=this._dom.target=target.nodeName?target:$(target).get(0);this._original.html=this._target.innerHTML;_instances.push(this);if(!this._target){Galleria.raise('Target not found',true);return}
this._options={autoplay:false,carousel:true,carouselFollow:true,carouselSpeed:400,carouselSteps:'auto',clicknext:false,dailymotion:{foreground:'%23EEEEEE',highlight:'%235BCEC5',background:'%23222222',logo:0,hideInfos:1},dataConfig: function(elem){return{}},dataSelector:'img',dataSort:false,dataSource:this._target,debug:undef,dummy:undef,easing:'galleria',extend: function(options){},fullscreenCrop:undef,fullscreenDoubleTap:true,fullscreenTransition:undef,height:0,idleMode:true,idleTime:3000,idleSpeed:200,imageCrop:false,imageMargin:0,imagePan:false,imagePanSmoothness:12,imagePosition:'50%',imageTimeout:undef,initialTransition:undef,keepSource:false,layerFollow:true,lightbox:false,lightboxFadeSpeed:200,lightboxTransitionSpeed:200,linkSourceImages:true,maxScaleRatio:undef,maxVideoSize:undef,minScaleRatio:undef,overlayOpacity:0.85,overlayBackground:'#0b0b0b',pauseOnInteraction:true,popupLinks:false,preload:2,queue:true,responsive:true,show:0,showInfo:true,showCounter:true,showImagenav:true,swipe:true,thumbCrop:true,thumbEventType:'click',thumbFit:true,thumbMargin:0,thumbQuality:'auto',thumbDisplayOrder:true,thumbnails:true,touchTransition:undef,transition:'fade',transitionInitial:undef,transitionSpeed:400,trueFullscreen:true,useCanvas:false,vimeo:{title:0,byline:0,portrait:0,color:'aaaaaa'},wait:5000,width:'auto',youtube:{modestbranding:1,autohide:1,color:'white',hd:1,rel:0,showinfo:0}};this._options.initialTransition=this._options.initialTransition||this._options.transitionInitial;if(options&&options.debug===false){DEBUG=false}
if(options&&typeof options.imageTimeout==='number'){TIMEOUT=options.imageTimeout}
if(options&&typeof options.dummy==='string'){DUMMY=options.dummy}
$(this._target).children().hide();if(Galleria.QUIRK){Galleria.raise('Your page is in Quirks mode, Galleria may not render correctly. Please validate your HTML and add a correct doctype.')}
if(typeof Galleria.theme==='object'){this._init()} else{_pool.push(this)}
return this},_init: function(){var self=this,options=this._options;if(this._initialized){Galleria.raise('Init failed: Gallery instance already initialized.');return this}
this._initialized=true;if(!Galleria.theme){Galleria.raise('Init failed: No theme found.',true);return this}
$.extend(true,options,Galleria.theme.defaults,this._original.options,Galleria.configure.options);(function(can){if(!('getContext' in can)){can=null;return}
_canvas=_canvas||{elem:can,context:can.getContext('2d'),cache:{},length:0}}(doc.createElement('canvas')));this.bind(Galleria.DATA, function(){this._original.data=this._data;this.get('total').innerHTML=this.getDataLength();var $container=this.$('container');if(self._options.height<2){self._userRatio=self._ratio=self._options.height}
var num={width:0,height:0};var testHeight=function(){return self.$('stage').height()};Utils.wait({until: function(){num=self._getWH();$container.width(num.width).height(num.height);return testHeight()&&num.width&&num.height>50},success: function(){self._width=num.width;self._height=num.height;self._ratio=self._ratio||num.height/num.width;if(Galleria.WEBKIT){window.setTimeout( function(){self._run()},1)} else{self._run()}},error: function(){if(testHeight()){Galleria.raise('Could not extract sufficient width/height of the gallery container. Traced measures: width:'+num.width+'px, height: '+num.height+'px.',true)} else{Galleria.raise('Could not extract a stage height from the CSS. Traced height: '+testHeight()+'px.',true)}},timeout:typeof this._options.wait=='number'?this._options.wait:false})});this.append({'info-text':['info-title','info-description'],'info':['info-text'],'image-nav':['image-nav-right','image-nav-left'],'stage':['images','loader','counter','image-nav'],'thumbnails-list':['thumbnails'],'thumbnails-container':['thumb-nav-left','thumbnails-list','thumb-nav-right'],'container':['stage','thumbnails-container','info','tooltip']});Utils.hide(this.$('counter').append(this.get('current'),doc.createTextNode(' / '),this.get('total')));this.setCounter('&#8211;');Utils.hide(self.get('tooltip'));this.$('container').addClass(Galleria.TOUCH?'touch':'notouch');$.each(new Array(2), function(i){var image=new Galleria.Picture();$(image.container).css({position:'absolute',top:0,left:0}).prepend(self._layers[i]=$(Utils.create('galleria-layer')).css({position:'absolute',top:0,left:0,right:0,bottom:0,zIndex:2})[0]);self.$('images').append(image.container);self._controls[i]=image});this.$('images').css({position:'relative',top:0,left:0,width:'100%',height:'100%'});this.$('thumbnails, thumbnails-list').css({overflow:'hidden',position:'relative'});this.$('image-nav-right, image-nav-left').bind('click', function(e){if(options.clicknext){e.stopPropagation()}
if(options.pauseOnInteraction){self.pause()}
var fn=/right/.test(this.className)?'next':'prev';self[fn]()});$.each(['info','counter','image-nav'], function(i,el){if(options['show'+el.substr(0,1).toUpperCase()+el.substr(1).replace(/-/,'')]===false){Utils.moveOut(self.get(el.toLowerCase()))}});this.load();if(!options.keepSource&&!IE){this._target.innerHTML=''}
if(this.get('errors')){this.appendChild('target','errors')}
this.appendChild('target','container');if(options.carousel){var count=0,show=options.show;this.bind(Galleria.THUMBNAIL, function(){this.updateCarousel();if(++count==this.getDataLength()&&typeof show=='number'&&show>0){this._carousel.follow(show)}})}
if(options.responsive){$win.bind('resize', function(){if(!self.isFullscreen()){self.resize()}})}
if(options.swipe){(function(images){var swipeStart=[0,0],swipeStop=[0,0],limitX=30,limitY=100,multi=false,tid=0,data,ev={start:'touchstart',move:'touchmove',stop:'touchend'},getData=function(e){return e.originalEvent.touches?e.originalEvent.touches[0]:e},moveHandler=function(e){if(e.originalEvent.touches&&e.originalEvent.touches.length>1){return}
data=getData(e);swipeStop=[data.pageX,data.pageY];if(!swipeStart[0]){swipeStart=swipeStop}
if(Math.abs(swipeStart[0]-swipeStop[0])>10){e.preventDefault()}},upHandler=function(e){images.unbind(ev.move,moveHandler);if((e.originalEvent.touches&&e.originalEvent.touches.length)||multi){multi=!multi;return}
if(Utils.timestamp()-tid<1000&&Math.abs(swipeStart[0]-swipeStop[0])>limitX&&Math.abs(swipeStart[1]-swipeStop[1])<limitY){e.preventDefault();self[swipeStart[0]>swipeStop[0]?'next':'prev']()}
swipeStart=swipeStop=[0,0]};images.bind(ev.start, function(e){if(e.originalEvent.touches&&e.originalEvent.touches.length>1){return}
data=getData(e);tid=Utils.timestamp();swipeStart=swipeStop=[data.pageX,data.pageY];images.bind(ev.move,moveHandler).one(ev.stop,upHandler)})}(self.$('images')));if(options.fullscreenDoubleTap){this.$('stage').bind('touchstart',(function(){var last,cx,cy,lx,ly,now,getData=function(e){return e.originalEvent.touches?e.originalEvent.touches[0]:e};return function(e){now=Galleria.utils.timestamp();cx=getData(e).pageX;cy=getData(e).pageY;if((now-last<500)&&(cx-lx<20)&&(cy-ly<20)){self.toggleFullscreen();e.preventDefault();self.$('stage').unbind('touchend',arguments.callee);return}
last=now;lx=cx;ly=cy}}()))}}
Utils.optimizeTouch(this.get('container'));$.each(Galleria.on.binds, function(i,bind){if($.inArray(bind.hash,self._binds)==-1){self.bind(bind.type,bind.callback)}});return this},addTimer: function(){this._timer.add.apply(this._timer,Utils.array(arguments));return this},clearTimer: function(){this._timer.clear.apply(this._timer,Utils.array(arguments));return this},_getWH: function(){var $container=this.$('container'),$target=this.$('target'),self=this,num={},arr;$.each(['width','height'], function(i,m){if(self._options[m]&&typeof self._options[m]==='number'){num[m]=self._options[m]} else{arr=[Utils.parseValue($container.css(m)),Utils.parseValue($target.css(m)),$container[m](),$target[m]()];if(!self['_'+m]){arr.splice(arr.length,Utils.parseValue($container.css('min-'+m)),Utils.parseValue($target.css('min-'+m)))}
num[m]=Math.max.apply(Math,arr)}});if(self._userRatio){num.height=num.width * self._userRatio}
return num},_createThumbnails: function(push){this.get('total').innerHTML=this.getDataLength();var src,thumb,data,special,$container,self=this,o=this._options,i=push?this._data.length-push.length:0,chunk=i,thumbchunk=[],loadindex=0,gif=IE<8?'http://upload.wikimedia.org/wikipedia/commons/c/c0/Blank.gif':'data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw%3D%3D',active=(function(){var a=self.$('thumbnails').find('.active');if(!a.length){return false}
return a.find('img').attr('src')}()),optval=typeof o.thumbnails==='string'?o.thumbnails.toLowerCase():null,getStyle=function(prop){return doc.defaultView&&doc.defaultView.getComputedStyle?doc.defaultView.getComputedStyle(thumb.container,null)[prop]:$container.css(prop)},fake=function(image,index,container){return function(){$(container).append(image);self.trigger({type:Galleria.THUMBNAIL,thumbTarget:image,index:index,galleriaData:self.getData(index)})}},onThumbEvent=function(e){if(o.pauseOnInteraction){self.pause()}
var index=$(e.currentTarget).data('index');if(self.getIndex()!==index){self.show(index)}
e.preventDefault()},thumbComplete=function(thumb,callback){$(thumb.container).css('visibility','visible');self.trigger({type:Galleria.THUMBNAIL,thumbTarget:thumb.image,index:thumb.data.order,galleriaData:self.getData(thumb.data.order)});if(typeof callback=='function'){callback.call(self,thumb)}},onThumbLoad=function(thumb,callback){thumb.scale({width:thumb.data.width,height:thumb.data.height,crop:o.thumbCrop,margin:o.thumbMargin,canvas:o.useCanvas,complete: function(thumb){var top=['left','top'],arr=['Width','Height'],m,css,data=self.getData(thumb.index),special=data.thumb.split(':');$.each(arr, function(i,measure){m=measure.toLowerCase();if((o.thumbCrop!==true||o.thumbCrop===m)&&o.thumbFit){css={};css[m]=thumb[m];$(thumb.container).css(css);css={};css[top[i]]=0;$(thumb.image).css(css)}
thumb['outer'+measure]=$(thumb.container)['outer'+measure](true)});Utils.toggleQuality(thumb.image,o.thumbQuality===true||(o.thumbQuality==='auto'&&thumb.original.width<thumb.width * 3));if(data.iframe&&special.length==2&&special[0] in _video){_video[special[0]].getThumb(special[1],(function(img){return function(src){img.src=src;thumbComplete(thumb,callback)}}(thumb.image)))} else if(o.thumbDisplayOrder&&!thumb.lazy){$.each(thumbchunk, function(i,th){if(i===loadindex&&th.ready&&!th.displayed){loadindex++;th.displayed=true;thumbComplete(th,callback);return}})} else{thumbComplete(thumb,callback)}}})};if(!push){this._thumbnails=[];this.$('thumbnails').empty()}
for(;this._data[i];i++){data=this._data[i];src=data.thumb||data.image;if((o.thumbnails===true||optval=='lazy')&&(data.thumb||data.image)){thumb=new Galleria.Picture(i);thumb.index=i;thumb.displayed=false;thumb.lazy=false;thumb.video=false;this.$('thumbnails').append(thumb.container);$container=$(thumb.container);$container.css('visibility','hidden');thumb.data={width:Utils.parseValue(getStyle('width')),height:Utils.parseValue(getStyle('height')),order:i,src:src};if(o.thumbFit&&o.thumbCrop!==true){$container.css({width:'auto',height:'auto'})} else{$container.css({width:thumb.data.width,height:thumb.data.height})}
special=src.split(':');if(special.length==2&&special[0] in _video){thumb.video=true;thumb.ready=true;thumb.load(gif,{height:thumb.data.height,width:thumb.data.height*1.25},onThumbLoad)} else if(optval=='lazy'){$container.addClass('lazy');thumb.lazy=true;thumb.load(gif,{height:thumb.data.height,width:thumb.data.width})} else{thumb.load(src,onThumbLoad)}
if(o.preload==='all'){thumb.preload(data.image)}} else if(data.iframe||optval==='empty'||optval==='numbers'){thumb={container:Utils.create('galleria-image'),image:Utils.create('img','span'),ready:true};if(optval==='numbers'){$(thumb.image).text(i+1)}
if(data.iframe){$(thumb.image).addClass('iframe')}
this.$('thumbnails').append(thumb.container);window.setTimeout((fake)(thumb.image,i,thumb.container),50+(i*20))} else{thumb={container:null,image:null}}
$(thumb.container).add(o.keepSource&&o.linkSourceImages?data.original:null).data('index',i).bind(o.thumbEventType,onThumbEvent).data('thumbload',onThumbLoad);if(active===src){$(thumb.container).addClass('active')}
this._thumbnails.push(thumb)}
thumbchunk=this._thumbnails.slice(chunk);return this},lazyLoad: function(index,complete){var arr=index.constructor==Array?index:[index],self=this,thumbnails=this.$('thumbnails').children().filter(function(){return $(this).data('lazy-src')}),loaded=0;$.each(arr, function(i,ind){if(ind>self._thumbnails.length-1){return}
var thumb=self._thumbnails[ind],data=thumb.data,special=data.src.split(':'),callback=function(){if(++loaded==arr.length&&typeof complete=='function'){complete.call(self)}},thumbload=$(thumb.container).data('thumbload');if(thumb.video){thumbload.call(self,thumb,callback)} else{thumb.load(data.src, function(thumb){thumbload.call(self,thumb,callback)})}});return this},lazyLoadChunks: function(size,delay){var len=this.getDataLength(),i=0,n=0,arr=[],temp=[],self=this;delay=delay||0;for(;i<len;i++){temp.push(i);if(++n==size||i==len-1){arr.push(temp);n=0;temp=[]}}
var init=function(wait){var a=arr.shift();if(a){window.setTimeout(function(){self.lazyLoad(a, function(){init(true)})},(delay&&wait)?delay:0)}};init(false);return this},_run: function(){var self=this;self._createThumbnails();Utils.wait({timeout:10000,until: function(){if(Galleria.OPERA){self.$('stage').css('display','inline-block')}
self._stageWidth=self.$('stage').width();self._stageHeight=self.$('stage').height();return(self._stageWidth&&self._stageHeight>50)},success: function(){_galleries.push(self);Utils.show(self.get('counter'));if(self._options.carousel){self._carousel.bindControls()}
if(self._options.autoplay){self.pause();if(typeof self._options.autoplay==='number'){self._playtime=self._options.autoplay}
self._playing=true}
if(self._firstrun){if(self._options.autoplay){self.trigger(Galleria.PLAY)}
if(typeof self._options.show==='number'){self.show(self._options.show)}
return}
self._firstrun=true;if(Galleria.History){Galleria.History.change(function(value){if(isNaN(value)){window.history.go(-1)} else{self.show(value,undef,true)}})}
self.trigger(Galleria.READY);Galleria.theme.init.call(self,self._options);$.each(Galleria.ready.callbacks, function(i,fn){if(typeof fn=='function'){fn.call(self,self._options)}});self._options.extend.call(self,self._options);if(/^[0-9]{1,4}$/.test(HASH)&&Galleria.History){self.show(HASH,undef,true)} else if(self._data[self._options.show]){self.show(self._options.show)}
if(self._options.autoplay){self.trigger(Galleria.PLAY)}},error: function(){Galleria.raise('Stage width or height is too small to show the gallery. Traced measures: width:'+self._stageWidth+'px, height: '+self._stageHeight+'px.',true)}})},load: function(source,selector,config){var self=this,o=this._options;this._data=[];this._thumbnails=[];this.$('thumbnails').empty();if(typeof selector==='function'){config=selector;selector=null}
source=source||o.dataSource;selector=selector||o.dataSelector;config=config||o.dataConfig;if(/^function Object/.test(source.constructor)){source=[source]}
if(source.constructor===Array){if(this.validate(source)){this._data=source} else{Galleria.raise('Load failed: JSON Array not valid.')}} else{selector+=',.video,.iframe';$(source).find(selector).each( function(i,elem){elem=$(elem);var data={},parent=elem.parent(),href=parent.attr('href'),rel=parent.attr('rel');if(href&&(elem[0].nodeName=='IMG'||elem.hasClass('video'))&&_videoTest(href)){data.video=href} else if(href&&elem.hasClass('iframe')){data.iframe=href} else{data.image=data.big=href}
if(rel){data.big=rel}
$.each('big title description link layer'.split(' '), function(i,val){if(elem.data(val)){data[val]=elem.data(val)}});self._data.push($.extend({title:elem.attr('title')||'',thumb:elem.attr('src'),image:elem.attr('src'),big:elem.attr('src'),description:elem.attr('alt')||'',link:elem.attr('longdesc'),original:elem.get(0)},data,config(elem)))})}
if(typeof o.dataSort=='function'){protoArray.sort.call(this._data,o.dataSort)} else if(o.dataSort=='random'){this._data.sort( function(){return Math.round(Math.random())-0.5})}
if(this.getDataLength()){this._parseData().trigger(Galleria.DATA)}
return this},_parseData: function(){var self=this,current;$.each(this._data, function(i,data){current=self._data[i];if('thumb' in data===false){current.thumb=data.image}
if(!'big' in data){current.big=data.image}
if('video' in data){var result=_videoTest(data.video);if(result){current.iframe=_video[result.provider].embed(result.id)+(function(){if(typeof self._options[result.provider]=='object'){var str='?',arr=[];$.each(self._options[result.provider], function(key,val){arr.push(key+'='+val)});if(result.provider=='youtube'){arr=['wmode=opaque'].concat(arr)}
return str+arr.join('&')}
return ''}());delete current.video;if(!('thumb' in current)||!current.thumb){current.thumb=result.provider+':'+result.id}}}});return this},destroy: function(){this.$('target').data('galleria',null);this.$('container').unbind('galleria');this.get('target').innerHTML=this._original.html;this.clearTimer();Utils.removeFromArray(_instances,this);Utils.removeFromArray(_galleries,this);return this},splice: function(){var self=this,args=Utils.array(arguments);window.setTimeout(function(){protoArray.splice.apply(self._data,args);self._parseData()._createThumbnails()},2);return self},push: function(){var self=this,args=Utils.array(arguments);if(args.length==1&&args[0].constructor==Array){args=args[0]}
window.setTimeout(function(){protoArray.push.apply(self._data,args);self._parseData()._createThumbnails(args)},2);return self},_getActive: function(){return this._controls.getActive()},validate: function(data){return true},bind: function(type,fn){type=_patchEvent(type);this.$('container').bind(type,this.proxy(fn));return this},unbind: function(type){type=_patchEvent(type);this.$('container').unbind(type);return this},trigger: function(type){type=typeof type==='object'?$.extend(type,{scope:this}):{type:_patchEvent(type),scope:this};this.$('container').trigger(type);return this},addIdleState: function(elem,styles,from,hide){this._idle.add.apply(this._idle,Utils.array(arguments));return this},removeIdleState: function(elem){this._idle.remove.apply(this._idle,Utils.array(arguments));return this},enterIdleMode: function(){this._idle.hide();return this},exitIdleMode: function(){this._idle.showAll();return this},enterFullscreen: function(callback){this._fullscreen.enter.apply(this,Utils.array(arguments));return this},exitFullscreen: function(callback){this._fullscreen.exit.apply(this,Utils.array(arguments));return this},toggleFullscreen: function(callback){this._fullscreen[this.isFullscreen()?'exit':'enter'].apply(this,Utils.array(arguments));return this},bindTooltip: function(elem,value){this._tooltip.bind.apply(this._tooltip,Utils.array(arguments));return this},defineTooltip: function(elem,value){this._tooltip.define.apply(this._tooltip,Utils.array(arguments));return this},refreshTooltip: function(elem){this._tooltip.show.apply(this._tooltip,Utils.array(arguments));return this},openLightbox: function(){this._lightbox.show.apply(this._lightbox,Utils.array(arguments));return this},closeLightbox: function(){this._lightbox.hide.apply(this._lightbox,Utils.array(arguments));return this},getActiveImage: function(){return this._getActive().image||undef},getActiveThumb: function(){return this._thumbnails[this._active].image||undef},getMousePosition: function(e){return{x:e.pageX-this.$('container').offset().left,y:e.pageY-this.$('container').offset().top}},addPan: function(img){if(this._options.imageCrop===false){return}
img=$(img||this.getActiveImage());var self=this,x=img.width()/2,y=img.height()/2,destX=parseInt(img.css('left'),10),destY=parseInt(img.css('top'),10),curX=destX||0,curY=destY||0,distX=0,distY=0,active=false,ts=Utils.timestamp(),cache=0,move=0,position=function(dist,cur,pos){if(dist>0){move=Math.round(Math.max(dist *-1,Math.min(0,cur)));if(cache!==move){cache=move;if(IE===8){img.parent()['scroll'+pos](move *-1)} else{var css={};css[pos.toLowerCase()]=move;img.css(css)}}}},calculate=function(e){if(Utils.timestamp()-ts<50){return}
active=true;x=self.getMousePosition(e).x;y=self.getMousePosition(e).y},loop=function(e){if(!active){return}
distX=img.width()-self._stageWidth;distY=img.height()-self._stageHeight;destX=x/self._stageWidth * distX *-1;destY=y/self._stageHeight * distY *-1;curX+=(destX-curX)/self._options.imagePanSmoothness;curY+=(destY-curY)/self._options.imagePanSmoothness;position(distY,curY,'Top');position(distX,curX,'Left')};if(IE===8){img.parent().scrollTop(curY *-1).scrollLeft(curX *-1);img.css({top:0,left:0})}
this.$('stage').unbind('mousemove',calculate).bind('mousemove',calculate);this.addTimer('pan'+self._id,loop,50,true);return this},proxy: function(fn,scope){if(typeof fn!=='function'){return F}
scope=scope||this;return function(){return fn.apply(scope,Utils.array(arguments))}},removePan: function(){this.$('stage').unbind('mousemove');this.clearTimer('pan'+this._id);return this},addElement: function(id){var dom=this._dom;$.each(Utils.array(arguments), function(i,blueprint){dom[blueprint]=Utils.create('galleria-'+blueprint)});return this},attachKeyboard: function(map){this._keyboard.attach.apply(this._keyboard,Utils.array(arguments));return this},detachKeyboard: function(){this._keyboard.detach.apply(this._keyboard,Utils.array(arguments));return this},appendChild: function(parentID,childID){this.$(parentID).append(this.get(childID)||childID);return this},prependChild: function(parentID,childID){this.$(parentID).prepend(this.get(childID)||childID);return this},remove: function(elemID){this.$(Utils.array(arguments).join(',')).remove();return this},append: function(data){var i,j;for(i in data){if(data.hasOwnProperty(i)){if(data[i].constructor===Array){for(j=0;data[i][j];j++){this.appendChild(i,data[i][j])}} else{this.appendChild(i,data[i])}}}
return this},_scaleImage: function(image,options){image=image||this._controls.getActive();if(!image){return}
var self=this,complete,scaleLayer=function(img){$(img.container).children(':first').css({top:Math.max(0,Utils.parseValue(img.image.style.top)),left:Math.max(0,Utils.parseValue(img.image.style.left)),width:Utils.parseValue(img.image.width),height:Utils.parseValue(img.image.height)})};options=$.extend({width:this._stageWidth,height:this._stageHeight,crop:this._options.imageCrop,max:this._options.maxScaleRatio,min:this._options.minScaleRatio,margin:this._options.imageMargin,position:this._options.imagePosition,iframelimit:this._options.maxVideoSize},options);if(this._options.layerFollow&&this._options.imageCrop!==true){if(typeof options.complete=='function'){complete=options.complete;options.complete=function(){complete.call(image,image);scaleLayer(image)}} else{options.complete=scaleLayer}} else{$(image.container).children(':first').css({top:0,left:0})}
image.scale(options);return this},updateCarousel: function(){this._carousel.update();return this},resize: function(measures,complete){if(typeof measures=='function'){complete=measures;measures=undef}
measures=$.extend({width:0,height:0},measures);var self=this,$container=this.$('container');$.each(measures, function(m,val){if(!val){$container[m]('auto');measures[m]=self._getWH()[m]}});$.each(measures, function(m,val){$container[m](val)});return this.rescale(complete)},rescale: function(width,height,complete){var self=this;if(typeof width==='function'){complete=width;width=undef}
var scale=function(){self._stageWidth=width||self.$('stage').width();self._stageHeight=height||self.$('stage').height();self._scaleImage();if(self._options.carousel){self.updateCarousel()}
self.trigger(Galleria.RESCALE);if(typeof complete==='function'){complete.call(self)}};scale.call(self);return this},refreshImage: function(){this._scaleImage();if(this._options.imagePan){this.addPan()}
return this},show: function(index,rewind,_history){if(this._queue.length>3||index===false||(!this._options.queue&&this._queue.stalled)){return}
index=Math.max(0,Math.min(parseInt(index,10),this.getDataLength()-1));rewind=typeof rewind!=='undefined'?!!rewind:index<this.getIndex();_history=_history||false;if(!_history&&Galleria.History){Galleria.History.set(index.toString());return}
this._active=index;protoArray.push.call(this._queue,{index:index,rewind:rewind});if(!this._queue.stalled){this._show()}
return this},_show: function(){var self=this,queue=this._queue[0],data=this.getData(queue.index);if(!data){return}
var src=data.iframe||(this.isFullscreen()&&'big' in data?data.big:data.image),active=this._controls.getActive(),next=this._controls.getNext(),cached=next.isCached(src),thumb=this._thumbnails[queue.index],mousetrigger=function(){$(next.image).trigger('mouseup')};var complete=(function(data,next,active,queue,thumb){return function(){var win;_transitions.active=false;Utils.toggleQuality(next.image,self._options.imageQuality);self._layers[self._controls.active].innerHTML='';$(active.container).css({zIndex:0,opacity:0}).show();if(active.isIframe){$(active.container).find('iframe').remove()}
self.$('container').toggleClass('iframe',!!data.iframe);$(next.container).css({zIndex:1,left:0,top:0}).show();self._controls.swap();if(self._options.imagePan){self.addPan(next.image)}
if(data.link||self._options.lightbox||self._options.clicknext){$(next.image).css({cursor:'pointer'}).bind('mouseup', function(e){if(typeof e.which=='number'&&e.which>1){return}
if(self._options.clicknext&&!Galleria.TOUCH){if(self._options.pauseOnInteraction){self.pause()}
self.next();return}
if(data.link){if(self._options.popupLinks){win=window.open(data.link,'_blank')} else{window.location.href=data.link}
return}
if(self._options.lightbox){self.openLightbox()}})}
self._playCheck();self.trigger({type:Galleria.IMAGE,index:queue.index,imageTarget:next.image,thumbTarget:thumb.image,galleriaData:data});protoArray.shift.call(self._queue);self._queue.stalled=false;if(self._queue.length){self._show()}}}(data,next,active,queue,thumb));if(this._options.carousel&&this._options.carouselFollow){this._carousel.follow(queue.index)}
if(this._options.preload){var p,i,n=this.getNext(),ndata;try{for(i=this._options.preload;i>0;i--){p=new Galleria.Picture();ndata=self.getData(n);p.preload(this.isFullscreen()&&'big' in ndata?ndata.big:ndata.image);n=self.getNext(n)}} catch(e){}}
Utils.show(next.container);next.isIframe=!!data.iframe;$(self._thumbnails[queue.index].container).addClass('active').siblings('.active').removeClass('active');self.trigger({type:Galleria.LOADSTART,cached:cached,index:queue.index,rewind:queue.rewind,imageTarget:next.image,thumbTarget:thumb.image,galleriaData:data});self._queue.stalled=true;next.load(src, function(next){var layer=$(self._layers[1-self._controls.active]).html(data.layer||'').hide();self._scaleImage(next,{complete: function(next){if('image' in active){Utils.toggleQuality(active.image,false)}
Utils.toggleQuality(next.image,false);self.removePan();self.setInfo(queue.index);self.setCounter(queue.index);if(data.layer){layer.show();if(data.link||self._options.lightbox||self._options.clicknext){layer.css('cursor','pointer').unbind('mouseup').mouseup(mousetrigger)}}
var transition=self._options.transition;$.each({initial:active.image===null,touch:Galleria.TOUCH,fullscreen:self.isFullscreen()}, function(type,arg){if(arg&&self._options[type+'Transition']!==undef){transition=self._options[type+'Transition'];return false}});if(transition in _transitions.effects===false){complete()} else{var params={prev:active.container,next:next.container,rewind:queue.rewind,speed:self._options.transitionSpeed||400};_transitions.active=true;_transitions.init.call(self,transition,params,complete)}
self.trigger({type:Galleria.LOADFINISH,cached:cached,index:queue.index,rewind:queue.rewind,imageTarget:next.image,thumbTarget:self._thumbnails[queue.index].image,galleriaData:self.getData(queue.index)})}})})},getNext: function(base){base=typeof base==='number'?base:this.getIndex();return base===this.getDataLength()-1?0:base+1},getPrev: function(base){base=typeof base==='number'?base:this.getIndex();return base===0?this.getDataLength()-1:base-1},next: function(){if(this.getDataLength()>1){this.show(this.getNext(),false)}
return this},prev: function(){if(this.getDataLength()>1){this.show(this.getPrev(),true)}
return this},get: function(elemId){return elemId in this._dom?this._dom[elemId]:null},getData: function(index){return index in this._data?this._data[index]:this._data[this._active]},getDataLength: function(){return this._data.length},getIndex: function(){return typeof this._active==='number'?this._active:false},getStageHeight: function(){return this._stageHeight},getStageWidth: function(){return this._stageWidth},getOptions: function(key){return typeof key==='undefined'?this._options:this._options[key]},setOptions: function(key,value){if(typeof key==='object'){$.extend(this._options,key)} else{this._options[key]=value}
return this},play: function(delay){this._playing=true;this._playtime=delay||this._playtime;this._playCheck();this.trigger(Galleria.PLAY);return this},pause: function(){this._playing=false;this.trigger(Galleria.PAUSE);return this},playToggle: function(delay){return(this._playing)?this.pause():this.play(delay)},isPlaying: function(){return this._playing},isFullscreen: function(){return this._fullscreen.active},_playCheck: function(){var self=this,played=0,interval=20,now=Utils.timestamp(),timer_id='play'+this._id;if(this._playing){this.clearTimer(timer_id);var fn=function(){played=Utils.timestamp()-now;if(played>=self._playtime&&self._playing){self.clearTimer(timer_id);self.next();return}
if(self._playing){self.trigger({type:Galleria.PROGRESS,percent:Math.ceil(played/self._playtime * 100),seconds:Math.floor(played/1000),milliseconds:played});self.addTimer(timer_id,fn,interval)}};self.addTimer(timer_id,fn,interval)}},setPlaytime: function(delay){this._playtime=delay;return this},setIndex: function(val){this._active=val;return this},setCounter: function(index){if(typeof index==='number'){index++} else if(typeof index==='undefined'){index=this.getIndex()+1}
this.get('current').innerHTML=index;if(IE){var count=this.$('counter'),opacity=count.css('opacity');if(parseInt(opacity,10)===1){Utils.removeAlpha(count[0])} else{this.$('counter').css('opacity',opacity)}}
return this},setInfo: function(index){var self=this,data=this.getData(index);$.each(['title','description'], function(i,type){var elem=self.$('info-'+type);if(!!data[type]){elem[data[type].length?'show':'hide']().html(data[type])} else{elem.empty().hide()}});return this},hasInfo: function(index){var check='title description'.split(' '),i;for(i=0;check[i];i++){if(!!this.getData(index)[check[i]]){return true}}
return false},jQuery: function(str){var self=this,ret=[];$.each(str.split(','), function(i,elemId){elemId=$.trim(elemId);if(self.get(elemId)){ret.push(elemId)}});var jQ=$(self.get(ret.shift()));$.each(ret, function(i,elemId){jQ=jQ.add(self.get(elemId))});return jQ},$: function(str){return this.jQuery.apply(this,Utils.array(arguments))}};$.each(_events, function(i,ev){var type=/_/.test(ev)?ev.replace(/_/g,''):ev;Galleria[ev.toUpperCase()]='galleria.'+type});$.extend(Galleria,{IE9:IE===9,IE8:IE===8,IE7:IE===7,IE6:IE===6,IE:IE,WEBKIT:/webkit/.test(NAV),CHROME:/chrome/.test(NAV),SAFARI:/safari/.test(NAV)&&!(/chrome/.test(NAV)),QUIRK:(IE&&doc.compatMode&&doc.compatMode==="BackCompat"),MAC:/mac/.test(navigator.platform.toLowerCase()),OPERA:!!window.opera,IPHONE:/iphone/.test(NAV),IPAD:/ipad/.test(NAV),ANDROID:/android/.test(NAV),TOUCH:('ontouchstart' in doc)});Galleria.addTheme=function(theme){if(!theme.name){Galleria.raise('No theme name specified')}
if(typeof theme.defaults!=='object'){theme.defaults={}} else{theme.defaults=_legacyOptions(theme.defaults)}
var css=false,reg;if(typeof theme.css==='string'){$('link').each(function(i,link){reg=new RegExp(theme.css.replace('\+\+resource\+\+','\\+\\+resource\\+\\+'));if(reg.test(link.href)){css=true;_themeLoad(theme);return false}});if(!css){$('script').each(function(i,script){reg=new RegExp('galleria\\.'+theme.name.toLowerCase()+'\\.');if(reg.test(script.src)){css=script.src.replace(/[^\/]*$/,'')+theme.css;window.setTimeout(function(){Utils.loadCSS(css,'galleria-theme', function(){_themeLoad(theme)})},1)}})}
if(!css){Galleria.raise('No theme CSS loaded')}} else{_themeLoad(theme)}
return theme};Galleria.loadTheme=function(src,options){if($('script').filter(function(){return $(this).attr('src')==src}).length){return}
var loaded=false,err;$(window).load( function(){if(!loaded){err=window.setTimeout(function(){if(!loaded&&!Galleria.theme){Galleria.raise("Galleria had problems loading theme at "+src+". Please check theme path or load manually.",true)}},20000)}});Galleria.unloadTheme();Utils.loadScript(src, function(){loaded=true;window.clearTimeout(err)});return Galleria};Galleria.unloadTheme=function(){if(typeof Galleria.theme=='object'){$('script').each(function(i,script){if(new RegExp('galleria\\.'+Galleria.theme.name+'\\.').test(script.src)){$(script).remove()}});Galleria.theme=undef}
return Galleria};Galleria.get=function(index){if(!!_instances[index]){return _instances[index]} else if(typeof index!=='number'){return _instances} else{Galleria.raise('Gallery index '+index+' not found')}};Galleria.configure=function(key,value){var opts={};if(typeof key=='string'&&value){opts[key]=value;key=opts} else{$.extend(opts,key)}
Galleria.configure.options=opts;$.each(Galleria.get(), function(i,instance){instance.setOptions(opts)});return Galleria};Galleria.configure.options={};Galleria.on=function(type,callback){if(!type){return}
callback=callback||F;var hash=type+callback.toString().replace(/\s/g,'')+Utils.timestamp();$.each(Galleria.get(), function(i,instance){instance._binds.push(hash);instance.bind(type,callback)});Galleria.on.binds.push({type:type,callback:callback,hash:hash});return Galleria};Galleria.on.binds=[];Galleria.run=function(selector,options){if($.isFunction(options)){options={extend:options}}
$(selector||'#galleria').galleria(options);return Galleria};Galleria.addTransition=function(name,fn){_transitions.effects[name]=fn;return Galleria};Galleria.utils=Utils;Galleria.log=function(){var args=Utils.array(arguments);if('console' in window&&'log' in window.console){try{return window.console.log.apply(window.console,args)} catch(e){$.each(args, function(){window.console.log(this)})}} else{return window.alert(args.join('<br>'))}};Galleria.ready=function(fn){if(typeof fn!='function'){return Galleria}
$.each(_galleries, function(i,gallery){fn.call(gallery,gallery._options)});Galleria.ready.callbacks.push(fn);return Galleria};Galleria.ready.callbacks=[];Galleria.raise=function(msg,fatal){var type=fatal?'Fatal error':'Error',self=this,css={color:'#fff',position:'absolute',top:0,left:0,zIndex:100000},echo=function(msg){var html='<div style="padding:4px;margin:0 0 2px;background:#'+(fatal?'811':'222')+';">'+(fatal?'<strong>'+type+': </strong>':'')+msg+'</div>';$.each(_instances, function(){var cont=this.$('errors'),target=this.$('target');if(!cont.length){target.css('position','relative');cont=this.addElement('errors').appendChild('target','errors').$('errors').css(css)}
cont.append(html)});if(!_instances.length){$('<div>').css($.extend(css,{position:'fixed'})).append(html).appendTo(DOM().body)}};if(DEBUG){echo(msg);if(fatal){throw new Error(type+': '+msg)}} else if(fatal){if(_hasError){return}
_hasError=true;fatal=false;echo('Gallery could not load.')}};Galleria.version=VERSION;Galleria.requires=function(version,msg){msg=msg||'You need to upgrade Galleria to version '+version+' to use one or more components.';if(Galleria.version<version){Galleria.raise(msg,true)}
return Galleria};Galleria.Picture=function(id){this.id=id||null;this.image=null;this.container=Utils.create('galleria-image');$(this.container).css({overflow:'hidden',position:'relative'});this.original={width:0,height:0};this.ready=false;this.isIframe=false};Galleria.Picture.prototype={cache:{},show: function(){Utils.show(this.image)},hide: function(){Utils.moveOut(this.image)},clear: function(){this.image=null},isCached: function(src){return!!this.cache[src]},preload: function(src){$(new Image()).load((function(src,cache){return function(){cache[src]=src}}(src,this.cache))).attr('src',src)},load: function(src,size,callback){if(typeof size=='function'){callback=size;size=null}
if(this.isIframe){var id='if'+new Date().getTime();this.image=$('<iframe>',{src:src,frameborder:0,id:id,allowfullscreen:true,css:{visibility:'hidden'}})[0];$(this.container).find('iframe,img').remove();this.container.appendChild(this.image);$('#'+id).load((function(self,callback){return function(){window.setTimeout(function(){$(self.image).css('visibility','visible');if(typeof callback=='function'){callback.call(self,self)}},10)}}(this,callback)));return this.container}
this.image=new Image();if(Galleria.IE8){$(this.image).css('filter','inherit')}
var i=0,reload=false,resort=false,$container=$(this.container),$image=$(this.image),onerror=function(){if(!reload){reload=true;window.setTimeout((function(image,src){return function(){image.attr('src',src+'?'+Utils.timestamp())}}($(this),src)),50)} else{if(DUMMY){$(this).attr('src',DUMMY)} else{Galleria.raise('Image not found: '+src)}}},onload=(function(self,callback,src){return function(){var complete=function(){$(this).unbind('load');self.original=size||{height:this.height,width:this.width};self.container.appendChild(this);self.cache[src]=src;if(typeof callback=='function'){window.setTimeout(function(){callback.call(self,self)},1)}};if((!this.width||!this.height)){window.setTimeout((function(img){return function(){if(img.width&&img.height){complete.call(img)} else{if(!resort){$(new Image()).load(onload).attr('src',img.src);resort=true} else{Galleria.raise('Could not extract width/height from image: '+img.src+'. Traced measures: width:'+img.width+'px, height: '+img.height+'px.')}}}}(this)),2)} else{complete.call(this)}}}(this,callback,src));$container.find('iframe,img').remove();$image.css('display','block');Utils.hide(this.image);$.each('minWidth minHeight maxWidth maxHeight'.split(' '), function(i,prop){$image.css(prop,(/min/.test(prop)?'0':'none'))});$image.load(onload).error(onerror).attr('src',src);return this.container},scale: function(options){var self=this;options=$.extend({width:0,height:0,min:undef,max:undef,margin:0,complete:F,position:'center',crop:false,canvas:false,iframelimit:undef},options);if(this.isIframe){var cw=options.width,ch=options.height,nw,nh;if(options.iframelimit){var r=Math.min(options.iframelimit/cw,options.iframelimit/ch);if(r<1){nw=cw * r;nh=ch * r;$(this.image).css({top:ch/2-nh/2,left:cw/2-nw/2,position:'absolute'})} else{$(this.image).css({top:0,left:0})}}
$(this.image).width(nw||cw).height(nh||ch).removeAttr('width').removeAttr('height');$(this.container).width(cw).height(ch);options.complete.call(self,self);try{if(this.image.contentWindow){$(this.image.contentWindow).trigger('resize')}} catch(e){}
return this.container}
if(!this.image){return this.container}
var width,height,$container=$(self.container),data;Utils.wait({until: function(){width=options.width||$container.width()||Utils.parseValue($container.css('width'));height=options.height||$container.height()||Utils.parseValue($container.css('height'));return width&&height},success: function(){var newWidth=(width-options.margin * 2)/self.original.width,newHeight=(height-options.margin * 2)/self.original.height,min=Math.min(newWidth,newHeight),max=Math.max(newWidth,newHeight),cropMap={'true':max,'width':newWidth,'height':newHeight,'false':min,'landscape':self.original.width>self.original.height?max:min,'portrait':self.original.width<self.original.height?max:min},ratio=cropMap[options.crop.toString()],canvasKey='';if(options.max){ratio=Math.min(options.max,ratio)}
if(options.min){ratio=Math.max(options.min,ratio)}
$.each(['width','height'], function(i,m){$(self.image)[m](self[m]=self.image[m]=Math.round(self.original[m] * ratio))});$(self.container).width(width).height(height);if(options.canvas&&_canvas){_canvas.elem.width=self.width;_canvas.elem.height=self.height;canvasKey=self.image.src+':'+self.width+'x'+self.height;self.image.src=_canvas.cache[canvasKey]||(function(key){_canvas.context.drawImage(self.image,0,0,self.original.width*ratio,self.original.height*ratio);try{data=_canvas.elem.toDataURL();_canvas.length+=data.length;_canvas.cache[key]=data;return data} catch(e){return self.image.src}}(canvasKey))}
var pos={},mix={},getPosition=function(value,measure,margin){var result=0;if (/\%/.test(value)){var flt=parseInt(value,10)/100,m=self.image[measure]||$(self.image)[measure]();result=Math.ceil(m *-1 * flt+margin * flt)} else{result=Utils.parseValue(value)}
return result},positionMap={'top':{top:0},'left':{left:0},'right':{left:'100%'},'bottom':{top:'100%'}};$.each(options.position.toLowerCase().split(' '), function(i,value){if(value==='center'){value='50%'}
pos[i?'top':'left']=value});$.each(pos, function(i,value){if(positionMap.hasOwnProperty(value)){$.extend(mix,positionMap[value])}});pos=pos.top?$.extend(pos,mix):mix;pos=$.extend({top:'50%',left:'50%'},pos);$(self.image).css({position:'absolute',top:getPosition(pos.top,'height',height),left:getPosition(pos.left,'width',width)});self.show();self.ready=true;options.complete.call(self,self)},error: function(){Galleria.raise('Could not scale image: '+self.image.src)},timeout:1000});return this}};$.extend($.easing,{galleria: function(_,t,b,c,d){if((t/=d/2)<1){return c/2*t*t*t+b}
return c/2*((t-=2)*t*t+2)+b},galleriaIn: function(_,t,b,c,d){return c*(t/=d)*t+b},galleriaOut: function(_,t,b,c,d){return-c *(t/=d)*(t-2)+b}});$.fn.galleria=function(options){var selector=this.selector;if(!$(this).length){$(function(){if($(selector).length){$(selector).galleria(options)} else{Galleria.utils.wait({until: function(){return $(selector).length},success: function(){$(selector).galleria(options)},error: function(){Galleria.raise('Init failed: Galleria could not find the element "'+selector+'".')},timeout:5000})}});return this}
return this.each(function(){if($.data(this,'galleria')){$.data(this,'galleria').destroy();$(this).find('*').hide()}
$.data(this,'galleria',new Galleria().init(this,options))})}}(jQuery));
