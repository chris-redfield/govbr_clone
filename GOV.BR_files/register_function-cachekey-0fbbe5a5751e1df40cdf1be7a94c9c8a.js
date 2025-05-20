
/* Merged Plone Javascript file
 * This file is dynamically assembled from separate parts.
 * Some of these parts have 3rd party licenses or copyright information attached
 * Such information is valid for that section,
 * not for the entire composite file
 * originating files are separated by - filename.js -
 */

/* - register_function.js - */
// https://www.gov.br/portal_javascripts/register_function.js?original=1
var bugRiddenCrashPronePieceOfJunk=(navigator.userAgent.indexOf('MSIE 5')!==-1&&navigator.userAgent.indexOf('Mac')!==-1);var W3CDOM=(!bugRiddenCrashPronePieceOfJunk&&typeof document.getElementsByTagName!=='undefined'&&typeof document.createElement!=='undefined');var registerEventListener=function(elem,event,func){jQuery(elem).bind(event,func)};var unRegisterEventListener=function(elem,event,func){jQuery(elem).unbind(event,func)};var registerPloneFunction=jQuery;
function getContentArea(){var node=jQuery('#region-content,#content');return node.length?node[0]:null}


/* - plone_javascript_variables.js - */
// https://www.gov.br/portal_javascripts/plone_javascript_variables.js?original=1
var portal_url='https://www.gov.br';var base_url='https://www.gov.br/pt-br/servicos/emitir-certidao-de-antecedentes-criminais';var form_modified_message='O seu formulário não foi salvo. Todas as alterações serão perdidas.';var form_resubmit_message='Você já clicou no botão de envio. Você realmente quer enviar este formulário novamente?';var external_links_open_new_window='false';var mark_special_links='false';var ajax_noresponse_message='Nenhuma resposta do servidor. Por favore, tente mais tarde.';

/* - ++resource++plone.app.jquerytools.js - */
!function($){function Overlay(trigger,conf){var closers,overlay,opened,self=this,fire=trigger.add(self),w=$(window),maskConf=$.tools.expose&&(conf.mask||conf.expose),uid=Math.random().toString().slice(10);maskConf&&("string"==typeof maskConf&&(maskConf={color:maskConf}),maskConf.closeOnClick=maskConf.closeOnEsc=!1);var jq=conf.target||trigger.attr("rel");if(overlay=jq?$(jq):trigger,!overlay.length)throw"Could not find Overlay: "+jq;trigger&&trigger.index(overlay)==-1&&trigger.click(function(e){return self.load(e),e.preventDefault()}),$.extend(self,{load:function(e){if(self.isOpened())return self;var eff=effects[conf.effect];if(!eff)throw'Overlay: cannot find effect : "'+conf.effect+'"';if(conf.oneInstance&&$.each(instances,function(){this.close(e)}),e=e||$.Event(),e.type="onBeforeLoad",fire.trigger(e),e.isDefaultPrevented())return self;opened=!0,maskConf&&$(overlay).expose(maskConf);var top=conf.top,left=conf.left,oWidth=overlay.outerWidth(!0),oHeight=overlay.outerHeight(!0);return"string"==typeof top&&(top="center"==top?Math.max((w.height()-oHeight)/2,0):parseInt(top,10)/100*w.height()),"center"==left&&(left=Math.max((w.width()-oWidth)/2,0)),eff[0].call(self,{top:top,left:left},function(){opened&&(e.type="onLoad",fire.trigger(e))}),maskConf&&conf.closeOnClick&&$.mask.getMask().one("click",self.close),conf.closeOnClick&&$(document).on("click."+uid,function(e){$(e.target).parents(overlay).length||self.close(e)}),conf.closeOnEsc&&$(document).on("keydown."+uid,function(e){27==e.keyCode&&self.close(e)}),self},close:function(e){return self.isOpened()?(e=e||$.Event(),e.type="onBeforeClose",fire.trigger(e),e.isDefaultPrevented()?void 0:(opened=!1,effects[conf.effect][1].call(self,function(){e.type="onClose",fire.trigger(e)}),$(document).off("click."+uid+" keydown."+uid),maskConf&&$.mask.close(),self)):self},getOverlay:function(){return overlay},getTrigger:function(){return trigger},getClosers:function(){return closers},isOpened:function(){return opened},getConf:function(){return conf}}),$.each("onBeforeLoad,onStart,onLoad,onBeforeClose,onClose".split(","),function(i,name){$.isFunction(conf[name])&&$(self).on(name,conf[name]),self[name]=function(fn){return fn&&$(self).on(name,fn),self}}),closers=overlay.find(conf.close||".close"),closers.length||conf.close||(closers=$('<a class="close"></a>'),overlay.prepend(closers)),closers.click(function(e){self.close(e)}),conf.load&&self.load()}$.tools=$.tools||{version:"@VERSION"},$.tools.overlay={addEffect:function(name,loadFn,closeFn){effects[name]=[loadFn,closeFn]},conf:{close:null,closeOnClick:!0,closeOnEsc:!0,closeSpeed:"fast",effect:"default",fixed:!/msie/.test(navigator.userAgent.toLowerCase())||navigator.appVersion>6,left:"center",load:!1,mask:null,oneInstance:!0,speed:"normal",target:null,top:"10%"}};var instances=[],effects={};$.tools.overlay.addEffect("default",function(pos,onLoad){var conf=this.getConf(),w=$(window);conf.fixed||(pos.top+=w.scrollTop(),pos.left+=w.scrollLeft()),pos.position=conf.fixed?"fixed":"absolute",this.getOverlay().css(pos).fadeIn(conf.speed,onLoad)},function(onClose){this.getOverlay().fadeOut(this.getConf().closeSpeed,onClose)}),$.fn.overlay=function(conf){var el=this.data("overlay");return el?el:($.isFunction(conf)&&(conf={onBeforeLoad:conf}),conf=$.extend(!0,{},$.tools.overlay.conf,conf),this.each(function(){el=new Overlay($(this),conf),instances.push(el),$(this).data("overlay",el)}),conf.api?el:this)}}(jQuery),function($){function find(root,query){var el=$(query);return el.length<2?el:root.parent().find(query)}function Scrollable(root,conf){var self=this,fire=root.add(self),itemWrap=root.children(),index=0,vertical=conf.vertical;if(current||(current=self),itemWrap.length>1&&(itemWrap=$(conf.items,root)),conf.size>1&&(conf.circular=!1),$.extend(self,{getConf:function(){return conf},getIndex:function(){return index},getSize:function(){return self.getItems().size()},getNaviButtons:function(){return prev.add(next)},getRoot:function(){return root},getItemWrap:function(){return itemWrap},getItems:function(){return itemWrap.find(conf.item).not("."+conf.clonedClass)},move:function(offset,time){return self.seekTo(index+offset,time)},next:function(time){return self.move(conf.size,time)},prev:function(time){return self.move(-conf.size,time)},begin:function(time){return self.seekTo(0,time)},end:function(time){return self.seekTo(self.getSize()-1,time)},focus:function(){return current=self,self},addItem:function(item){return item=$(item),conf.circular?(itemWrap.children().last().before(item),itemWrap.children().first().replaceWith(item.clone().addClass(conf.clonedClass))):(itemWrap.append(item),next.removeClass("disabled")),fire.trigger("onAddItem",[item]),self},seekTo:function(i,time,fn){if(i.jquery||(i*=1),conf.circular&&0===i&&index==-1&&0!==time)return self;if(!conf.circular&&i<0||i>self.getSize()||i<-1)return self;var item=i;i.jquery?i=self.getItems().index(i):item=self.getItems().eq(i);var e=$.Event("onBeforeSeek");if(!fn&&(fire.trigger(e,[i,time]),e.isDefaultPrevented()||!item.length))return self;var props=vertical?{top:-item.position().top}:{left:-item.position().left};return index=i,current=self,void 0===time&&(time=conf.speed),itemWrap.animate(props,time,conf.easing,fn||function(){fire.trigger("onSeek",[i])}),self}}),$.each(["onBeforeSeek","onSeek","onAddItem"],function(i,name){$.isFunction(conf[name])&&$(self).on(name,conf[name]),self[name]=function(fn){return fn&&$(self).on(name,fn),self}}),conf.circular){var cloned1=self.getItems().slice(-1).clone().prependTo(itemWrap),cloned2=self.getItems().eq(1).clone().appendTo(itemWrap);cloned1.add(cloned2).addClass(conf.clonedClass),self.onBeforeSeek(function(e,i,time){if(!e.isDefaultPrevented())return i==-1?(self.seekTo(cloned1,time,function(){self.end(0)}),e.preventDefault()):void(i==self.getSize()&&self.seekTo(cloned2,time,function(){self.begin(0)}))});var hidden_parents=root.parents().add(root).filter(function(){if("none"===$(this).css("display"))return!0});hidden_parents.length?(hidden_parents.show(),self.seekTo(0,0,function(){}),hidden_parents.hide()):self.seekTo(0,0,function(){})}var prev=find(root,conf.prev).click(function(e){e.stopPropagation(),self.prev()}),next=find(root,conf.next).click(function(e){e.stopPropagation(),self.next()});if(conf.circular||(self.onBeforeSeek(function(e,i){setTimeout(function(){e.isDefaultPrevented()||(prev.toggleClass(conf.disabledClass,i<=0),next.toggleClass(conf.disabledClass,i>=self.getSize()-1))},1)}),conf.initialIndex||prev.addClass(conf.disabledClass)),self.getSize()<2&&prev.add(next).addClass(conf.disabledClass),conf.mousewheel&&$.fn.mousewheel&&root.mousewheel(function(e,delta){if(conf.mousewheel)return self.move(delta<0?1:-1,conf.wheelSpeed||50),!1}),conf.touch){var touch={};itemWrap[0].ontouchstart=function(e){var t=e.touches[0];touch.x=t.clientX,touch.y=t.clientY},itemWrap[0].ontouchmove=function(e){if(1==e.touches.length&&!itemWrap.is(":animated")){var t=e.touches[0],deltaX=touch.x-t.clientX,deltaY=touch.y-t.clientY;self[vertical&&deltaY>0||!vertical&&deltaX>0?"next":"prev"](),e.preventDefault()}}}conf.keyboard&&$(document).on("keydown.scrollable",function(evt){if(!(!conf.keyboard||evt.altKey||evt.ctrlKey||evt.metaKey||$(evt.target).is(":input")||"static"!=conf.keyboard&&current!=self)){var key=evt.keyCode;return!vertical||38!=key&&40!=key?vertical||37!=key&&39!=key?void 0:(self.move(37==key?-1:1),evt.preventDefault()):(self.move(38==key?-1:1),evt.preventDefault())}}),conf.initialIndex&&self.seekTo(conf.initialIndex,0,function(){})}$.tools=$.tools||{version:"@VERSION"},$.tools.scrollable={conf:{activeClass:"active",circular:!1,clonedClass:"cloned",disabledClass:"disabled",easing:"swing",initialIndex:0,item:"> *",items:".items",keyboard:!0,mousewheel:!1,next:".next",prev:".prev",size:1,speed:400,vertical:!1,touch:!0,wheelSpeed:0}};var current;$.fn.scrollable=function(conf){var el=this.data("scrollable");return el?el:(conf=$.extend({},$.tools.scrollable.conf,conf),this.each(function(){el=new Scrollable($(this),conf),$(this).data("scrollable",el)}),conf.api?el:this)}}(jQuery),function($){function Tabs(root,paneSelector,conf){var current,self=this,trigger=root.add(this),tabs=root.find(conf.tabs),panes=paneSelector.jquery?paneSelector:root.children(paneSelector);tabs.length||(tabs=root.children()),panes.length||(panes=root.parent().find(paneSelector)),panes.length||(panes=$(paneSelector)),$.extend(this,{click:function(i,e){var tab=tabs.eq(i),firstRender=!root.data("tabs");if("string"==typeof i&&i.replace("#","")&&(tab=tabs.filter('[href*="'+i.replace("#","")+'"]'),i=Math.max(tabs.index(tab),0)),conf.rotate){var last=tabs.length-1;if(i<0)return self.click(last,e);if(i>last)return self.click(0,e)}if(!tab.length){if(current>=0)return self;i=conf.initialIndex,tab=tabs.eq(i)}if(i===current)return self;if(e=e||$.Event(),e.type="onBeforeClick",trigger.trigger(e,[i]),!e.isDefaultPrevented()){var effect=firstRender?conf.initialEffect&&conf.effect||"default":conf.effect;return effects[effect].call(self,i,function(){current=i,e.type="onClick",trigger.trigger(e,[i])}),tabs.removeClass(conf.current),tab.addClass(conf.current),self}},getConf:function(){return conf},getTabs:function(){return tabs},getPanes:function(){return panes},getCurrentPane:function(){return panes.eq(current)},getCurrentTab:function(){return tabs.eq(current)},getIndex:function(){return current},next:function(){return self.click(current+1)},prev:function(){return self.click(current-1)},destroy:function(){return tabs.off(conf.event).removeClass(conf.current),panes.find('a[href^="#"]').off("click.T"),self}}),$.each("onBeforeClick,onClick".split(","),function(i,name){$.isFunction(conf[name])&&$(self).on(name,conf[name]),self[name]=function(fn){return fn&&$(self).on(name,fn),self}}),conf.history&&$.fn.history&&($.tools.history.init(tabs),conf.event="history"),tabs.each(function(i){$(this).on(conf.event,function(e){return self.click(i,e),e.preventDefault()})}),panes.find('a[href^="#"]').on("click.T",function(e){self.click($(this).attr("href"),e)}),location.hash&&"a"==conf.tabs&&root.find('[href="'+location.hash+'"]').length?self.click(location.hash):(0===conf.initialIndex||conf.initialIndex>0)&&self.click(conf.initialIndex)}$.tools=$.tools||{version:"@VERSION"},$.tools.tabs={conf:{tabs:"a",current:"current",onBeforeClick:null,onClick:null,effect:"default",initialEffect:!1,initialIndex:0,event:"click",rotate:!1,slideUpSpeed:400,slideDownSpeed:400,history:!1},addEffect:function(name,fn){effects[name]=fn}};var animating,w,effects={default:function(i,done){this.getPanes().hide().eq(i).show(),done.call()},fade:function(i,done){var conf=this.getConf(),speed=conf.fadeOutSpeed,panes=this.getPanes();speed?panes.fadeOut(speed):panes.hide(),panes.eq(i).fadeIn(conf.fadeInSpeed,done)},slide:function(i,done){var conf=this.getConf();this.getPanes().slideUp(conf.slideUpSpeed),this.getPanes().eq(i).slideDown(conf.slideDownSpeed,done)},ajax:function(i,done){this.getPanes().eq(0).load(this.getTabs().eq(i).attr("href"),done)}};$.tools.tabs.addEffect("horizontal",function(i,done){if(!animating){var nextPane=this.getPanes().eq(i),currentPane=this.getCurrentPane();w||(w=this.getPanes().eq(0).width()),animating=!0,nextPane.show(),currentPane.animate({width:0},{step:function(now){nextPane.css("width",w-now)},complete:function(){$(this).hide(),done.call(),animating=!1}}),currentPane.length||(done.call(),animating=!1)}}),$.fn.tabs=function(paneSelector,conf){var el=this.data("tabs");return el&&(el.destroy(),this.removeData("tabs")),$.isFunction(conf)&&(conf={onBeforeClick:conf}),conf=$.extend({},$.tools.tabs.conf,conf),this.each(function(){el=new Tabs($(this),paneSelector,conf),$(this).data("tabs",el)}),conf.api?el:this}}(jQuery),function($){function setIframeLocation(h){if(h){var doc=iframe.contentWindow.document;doc.open().close(),doc.location.hash=h}}var hash,iframe,links,inited;$.tools=$.tools||{version:"@VERSION"},$.tools.history={init:function(els){inited||($.browser.msie&&$.browser.version<"8"?iframe||(iframe=$("<iframe/>").attr("src","javascript:false;").hide().get(0),$("body").append(iframe),setInterval(function(){var idoc=iframe.contentWindow.document,h=idoc.location.hash;hash!==h&&$(window).trigger("hash",h)},100),setIframeLocation(location.hash||"#")):setInterval(function(){var h=location.hash;h!==hash&&$(window).trigger("hash",h)},100),links=links?links.add(els):els,els.click(function(e){var href=$(this).attr("href");if(iframe&&setIframeLocation(href),"#"!=href.slice(0,1))return location.href="#"+href,e.preventDefault()}),inited=!0)}},$(window).on("hash",function(e,h){h?links.filter(function(){var href=$(this).attr("href");return href==h||href==h.replace("#","")}).trigger("history",[h]):links.eq(0).trigger("history",[h]),hash=h}),$.fn.history=function(fn){return $.tools.history.init(this),this.on("history",fn)}}(jQuery),function($){function viewport(){if(/msie/.test(navigator.userAgent.toLowerCase())){var d=$(document).height(),w=$(window).height();return[window.innerWidth||document.documentElement.clientWidth||document.body.clientWidth,d-w<20?w:d]}return[$(document).width(),$(document).height()]}function call(fn){if(fn)return fn.call($.mask)}$.tools=$.tools||{version:"@VERSION"};var tool;tool=$.tools.expose={conf:{maskId:"exposeMask",loadSpeed:"slow",closeSpeed:"fast",closeOnClick:!0,closeOnEsc:!0,zIndex:9998,opacity:.8,startOpacity:0,color:"#fff",onLoad:null,onClose:null}};var mask,exposed,loaded,config,overlayIndex;$.mask={load:function(conf,els){if(loaded)return this;"string"==typeof conf&&(conf={color:conf}),conf=conf||config,config=conf=$.extend($.extend({},tool.conf),conf),mask=$("#"+conf.maskId),mask.length||(mask=$("<div/>").attr("id",conf.maskId),$("body").append(mask));var size=viewport();return mask.css({position:"absolute",top:0,left:0,width:size[0],height:size[1],display:"none",opacity:conf.startOpacity,zIndex:conf.zIndex}),conf.color&&mask.css("backgroundColor",conf.color),call(conf.onBeforeLoad)===!1?this:(conf.closeOnEsc&&$(document).on("keydown.mask",function(e){27==e.keyCode&&$.mask.close(e)}),conf.closeOnClick&&mask.on("click.mask",function(e){$.mask.close(e)}),$(window).on("resize.mask",function(){$.mask.fit()}),els&&els.length&&(overlayIndex=els.eq(0).css("zIndex"),$.each(els,function(){var el=$(this);/relative|absolute|fixed/i.test(el.css("position"))||el.css("position","relative")}),exposed=els.css({zIndex:Math.max(conf.zIndex+1,"auto"==overlayIndex?0:overlayIndex)})),mask.css({display:"block"}).fadeTo(conf.loadSpeed,conf.opacity,function(){$.mask.fit(),call(conf.onLoad),loaded="full"}),loaded=!0,this)},close:function(){if(loaded){if(call(config.onBeforeClose)===!1)return this;mask.fadeOut(config.closeSpeed,function(){exposed&&exposed.css({zIndex:overlayIndex}),loaded=!1,call(config.onClose)}),$(document).off("keydown.mask"),mask.off("click.mask"),$(window).off("resize.mask")}return this},fit:function(){if(loaded){var size=viewport();mask.css({width:size[0],height:size[1]})}},getMask:function(){return mask},isLoaded:function(fully){return fully?"full"==loaded:loaded},getConf:function(){return config},getExposed:function(){return exposed}},$.fn.mask=function(conf){return $.mask.load(conf),this},$.fn.expose=function(conf){return $.mask.load(conf,this),this}}(jQuery);

/* - ++resource++plone.app.jquerytools.form.js - */
/*!
 * jQuery Form Plugin
 * version: 3.51.0-2014.06.20
 * Requires jQuery v1.5 or later
 * Copyright (c) 2014 M. Alsup
 * Examples and documentation at: http://malsup.com/jquery/form/
 * Project repository: https://github.com/malsup/form
 * Dual licensed under the MIT and GPL licenses.
 * https://github.com/malsup/form#copyright-and-license
 */
/*global ActiveXObject */

// AMD support
(function (factory) {
    "use strict";
    if (typeof define === 'function' && define.amd) {
        // using AMD; register as anon module
        define(['jquery'], factory);
    } else {
        // no AMD; invoke directly
        factory( (typeof(jQuery) != 'undefined') ? jQuery : window.Zepto );
    }
}

(function($) {
"use strict";

/*
    Usage Note:
    -----------
    Do not use both ajaxSubmit and ajaxForm on the same form.  These
    functions are mutually exclusive.  Use ajaxSubmit if you want
    to bind your own submit handler to the form.  For example,

    $(document).ready(function() {
        $('#myForm').on('submit', function(e) {
            e.preventDefault(); // <-- important
            $(this).ajaxSubmit({
                target: '#output'
            });
        });
    });

    Use ajaxForm when you want the plugin to manage all the event binding
    for you.  For example,

    $(document).ready(function() {
        $('#myForm').ajaxForm({
            target: '#output'
        });
    });

    You can also use ajaxForm with delegation (requires jQuery v1.7+), so the
    form does not have to exist when you invoke ajaxForm:

    $('#myForm').ajaxForm({
        delegation: true,
        target: '#output'
    });

    When using ajaxForm, the ajaxSubmit function will be invoked for you
    at the appropriate time.
*/

/**
 * Feature detection
 */
var feature = {};
feature.fileapi = $("<input type='file'/>").get(0).files !== undefined;
feature.formdata = window.FormData !== undefined;

var hasProp = !!$.fn.prop;

// attr2 uses prop when it can but checks the return type for
// an expected string.  this accounts for the case where a form 
// contains inputs with names like "action" or "method"; in those
// cases "prop" returns the element
$.fn.attr2 = function() {
    if ( ! hasProp ) {
        return this.attr.apply(this, arguments);
    }
    var val = this.prop.apply(this, arguments);
    if ( ( val && val.jquery ) || typeof val === 'string' ) {
        return val;
    }
    return this.attr.apply(this, arguments);
};

/**
 * ajaxSubmit() provides a mechanism for immediately submitting
 * an HTML form using AJAX.
 */
$.fn.ajaxSubmit = function(options) {
    /*jshint scripturl:true */

    // fast fail if nothing selected (http://dev.jquery.com/ticket/2752)
    if (!this.length) {
        log('ajaxSubmit: skipping submit process - no element selected');
        return this;
    }

    var method, action, url, $form = this;

    if (typeof options == 'function') {
        options = { success: options };
    }
    else if ( options === undefined ) {
        options = {};
    }

    method = options.type || this.attr2('method');
    action = options.url  || this.attr2('action');

    url = (typeof action === 'string') ? $.trim(action) : '';
    url = url || window.location.href || '';
    if (url) {
        // clean url (don't include hash vaue)
        url = (url.match(/^([^#]+)/)||[])[1];
    }

    options = $.extend(true, {
        url:  url,
        success: $.ajaxSettings.success,
        type: method || $.ajaxSettings.type,
        iframeSrc: /^https/i.test(window.location.href || '') ? 'javascript:false' : 'about:blank'
    }, options);

    // hook for manipulating the form data before it is extracted;
    // convenient for use with rich editors like tinyMCE or FCKEditor
    var veto = {};
    this.trigger('form-pre-serialize', [this, options, veto]);
    if (veto.veto) {
        log('ajaxSubmit: submit vetoed via form-pre-serialize trigger');
        return this;
    }

    // provide opportunity to alter form data before it is serialized
    if (options.beforeSerialize && options.beforeSerialize(this, options) === false) {
        log('ajaxSubmit: submit aborted via beforeSerialize callback');
        return this;
    }

    var traditional = options.traditional;
    if ( traditional === undefined ) {
        traditional = $.ajaxSettings.traditional;
    }

    var elements = [];
    var qx, a = this.formToArray(options.semantic, elements);
    if (options.data) {
        options.extraData = options.data;
        qx = $.param(options.data, traditional);
    }

    // give pre-submit callback an opportunity to abort the submit
    if (options.beforeSubmit && options.beforeSubmit(a, this, options) === false) {
        log('ajaxSubmit: submit aborted via beforeSubmit callback');
        return this;
    }

    // fire vetoable 'validate' event
    this.trigger('form-submit-validate', [a, this, options, veto]);
    if (veto.veto) {
        log('ajaxSubmit: submit vetoed via form-submit-validate trigger');
        return this;
    }

    var q = $.param(a, traditional);
    if (qx) {
        q = ( q ? (q + '&' + qx) : qx );
    }
    if (options.type.toUpperCase() == 'GET') {
        options.url += (options.url.indexOf('?') >= 0 ? '&' : '?') + q;
        options.data = null;  // data is null for 'get'
    }
    else {
        options.data = q; // data is the query string for 'post'
    }

    var callbacks = [];
    if (options.resetForm) {
        callbacks.push(function() { $form.resetForm(); });
    }
    if (options.clearForm) {
        callbacks.push(function() { $form.clearForm(options.includeHidden); });
    }

    // perform a load on the target only if dataType is not provided
    if (!options.dataType && options.target) {
        var oldSuccess = options.success || function(){};
        callbacks.push(function(data) {
            var fn = options.replaceTarget ? 'replaceWith' : 'html';
            $(options.target)[fn](data).each(oldSuccess, arguments);
        });
    }
    else if (options.success) {
        callbacks.push(options.success);
    }

    options.success = function(data, status, xhr) { // jQuery 1.4+ passes xhr as 3rd arg
        var context = options.context || this ;    // jQuery 1.4+ supports scope context
        for (var i=0, max=callbacks.length; i < max; i++) {
            callbacks[i].apply(context, [data, status, xhr || $form, $form]);
        }
    };

    if (options.error) {
        var oldError = options.error;
        options.error = function(xhr, status, error) {
            var context = options.context || this;
            oldError.apply(context, [xhr, status, error, $form]);
        };
    }

     if (options.complete) {
        var oldComplete = options.complete;
        options.complete = function(xhr, status) {
            var context = options.context || this;
            oldComplete.apply(context, [xhr, status, $form]);
        };
    }

    // are there files to upload?

    // [value] (issue #113), also see comment:
    // https://github.com/malsup/form/commit/588306aedba1de01388032d5f42a60159eea9228#commitcomment-2180219
    var fileInputs = $('input[type=file]:enabled', this).filter(function() { return $(this).val() !== ''; });

    var hasFileInputs = fileInputs.length > 0;
    var mp = 'multipart/form-data';
    var multipart = ($form.attr('enctype') == mp || $form.attr('encoding') == mp);

    var fileAPI = feature.fileapi && feature.formdata;
    log("fileAPI :" + fileAPI);
    var shouldUseFrame = (hasFileInputs || multipart) && !fileAPI;

    var jqxhr;

    // options.iframe allows user to force iframe mode
    // 06-NOV-09: now defaulting to iframe mode if file input is detected
    if (options.iframe !== false && (options.iframe || shouldUseFrame)) {
        // hack to fix Safari hang (thanks to Tim Molendijk for this)
        // see:  http://groups.google.com/group/jquery-dev/browse_thread/thread/36395b7ab510dd5d
        if (options.closeKeepAlive) {
            $.get(options.closeKeepAlive, function() {
                jqxhr = fileUploadIframe(a);
            });
        }
        else {
            jqxhr = fileUploadIframe(a);
        }
    }
    else if ((hasFileInputs || multipart) && fileAPI) {
        jqxhr = fileUploadXhr(a);
    }
    else {
        jqxhr = $.ajax(options);
    }

    $form.removeData('jqxhr').data('jqxhr', jqxhr);

    // clear element array
    for (var k=0; k < elements.length; k++) {
        elements[k] = null;
    }

    // fire 'notify' event
    this.trigger('form-submit-notify', [this, options]);
    return this;

    // utility fn for deep serialization
    function deepSerialize(extraData){
        var serialized = $.param(extraData, options.traditional).split('&');
        var len = serialized.length;
        var result = [];
        var i, part;
        for (i=0; i < len; i++) {
            // #252; undo param space replacement
            serialized[i] = serialized[i].replace(/\+/g,' ');
            part = serialized[i].split('=');
            // #278; use array instead of object storage, favoring array serializations
            result.push([decodeURIComponent(part[0]), decodeURIComponent(part[1])]);
        }
        return result;
    }

     // XMLHttpRequest Level 2 file uploads (big hat tip to francois2metz)
    function fileUploadXhr(a) {
        var formdata = new FormData();

        for (var i=0; i < a.length; i++) {
            formdata.append(a[i].name, a[i].value);
        }

        if (options.extraData) {
            var serializedData = deepSerialize(options.extraData);
            for (i=0; i < serializedData.length; i++) {
                if (serializedData[i]) {
                    formdata.append(serializedData[i][0], serializedData[i][1]);
                }
            }
        }

        options.data = null;

        var s = $.extend(true, {}, $.ajaxSettings, options, {
            contentType: false,
            processData: false,
            cache: false,
            type: method || 'POST'
        });

        if (options.uploadProgress) {
            // workaround because jqXHR does not expose upload property
            s.xhr = function() {
                var xhr = $.ajaxSettings.xhr();
                if (xhr.upload) {
                    xhr.upload.addEventListener('progress', function(event) {
                        var percent = 0;
                        var position = event.loaded || event.position; /*event.position is deprecated*/
                        var total = event.total;
                        if (event.lengthComputable) {
                            percent = Math.ceil(position / total * 100);
                        }
                        options.uploadProgress(event, position, total, percent);
                    }, false);
                }
                return xhr;
            };
        }

        s.data = null;
        var beforeSend = s.beforeSend;
        s.beforeSend = function(xhr, o) {
            //Send FormData() provided by user
            if (options.formData) {
                o.data = options.formData;
            }
            else {
                o.data = formdata;
            }
            if(beforeSend) {
                beforeSend.call(this, xhr, o);
            }
        };
        return $.ajax(s);
    }

    // private function for handling file uploads (hat tip to YAHOO!)
    function fileUploadIframe(a) {
        var form = $form[0], el, i, s, g, id, $io, io, xhr, sub, n, timedOut, timeoutHandle;
        var deferred = $.Deferred();

        // #341
        deferred.abort = function(status) {
            xhr.abort(status);
        };

        if (a) {
            // ensure that every serialized input is still enabled
            for (i=0; i < elements.length; i++) {
                el = $(elements[i]);
                if ( hasProp ) {
                    el.prop('disabled', false);
                }
                else {
                    el.removeAttr('disabled');
                }
            }
        }

        s = $.extend(true, {}, $.ajaxSettings, options);
        s.context = s.context || s;
        id = 'jqFormIO' + (new Date().getTime());
        if (s.iframeTarget) {
            $io = $(s.iframeTarget);
            n = $io.attr2('name');
            if (!n) {
                $io.attr2('name', id);
            }
            else {
                id = n;
            }
        }
        else {
            $io = $('<iframe name="' + id + '" src="'+ s.iframeSrc +'" />');
            $io.css({ position: 'absolute', top: '-1000px', left: '-1000px' });
        }
        io = $io[0];


        xhr = { // mock object
            aborted: 0,
            responseText: null,
            responseXML: null,
            status: 0,
            statusText: 'n/a',
            getAllResponseHeaders: function() {},
            getResponseHeader: function() {},
            setRequestHeader: function() {},
            abort: function(status) {
                var e = (status === 'timeout' ? 'timeout' : 'aborted');
                log('aborting upload... ' + e);
                this.aborted = 1;

                try { // #214, #257
                    if (io.contentWindow.document.execCommand) {
                        io.contentWindow.document.execCommand('Stop');
                    }
                }
                catch(ignore) {}

                $io.attr('src', s.iframeSrc); // abort op in progress
                xhr.error = e;
                if (s.error) {
                    s.error.call(s.context, xhr, e, status);
                }
                if (g) {
                    $.event.trigger("ajaxError", [xhr, s, e]);
                }
                if (s.complete) {
                    s.complete.call(s.context, xhr, e);
                }
            }
        };

        g = s.global;
        // trigger ajax global events so that activity/block indicators work like normal
        if (g && 0 === $.active++) {
            $.event.trigger("ajaxStart");
        }
        if (g) {
            $.event.trigger("ajaxSend", [xhr, s]);
        }

        if (s.beforeSend && s.beforeSend.call(s.context, xhr, s) === false) {
            if (s.global) {
                $.active--;
            }
            deferred.reject();
            return deferred;
        }
        if (xhr.aborted) {
            deferred.reject();
            return deferred;
        }

        // add submitting element to data if we know it
        sub = form.clk;
        if (sub) {
            n = sub.name;
            if (n && !sub.disabled) {
                s.extraData = s.extraData || {};
                s.extraData[n] = sub.value;
                if (sub.type == "image") {
                    s.extraData[n+'.x'] = form.clk_x;
                    s.extraData[n+'.y'] = form.clk_y;
                }
            }
        }

        var CLIENT_TIMEOUT_ABORT = 1;
        var SERVER_ABORT = 2;
                
        function getDoc(frame) {
            /* it looks like contentWindow or contentDocument do not
             * carry the protocol property in ie8, when running under ssl
             * frame.document is the only valid response document, since
             * the protocol is know but not on the other two objects. strange?
             * "Same origin policy" http://en.wikipedia.org/wiki/Same_origin_policy
             */
            
            var doc = null;
            
            // IE8 cascading access check
            try {
                if (frame.contentWindow) {
                    doc = frame.contentWindow.document;
                }
            } catch(err) {
                // IE8 access denied under ssl & missing protocol
                log('cannot get iframe.contentWindow document: ' + err);
            }

            if (doc) { // successful getting content
                return doc;
            }

            try { // simply checking may throw in ie8 under ssl or mismatched protocol
                doc = frame.contentDocument ? frame.contentDocument : frame.document;
            } catch(err) {
                // last attempt
                log('cannot get iframe.contentDocument: ' + err);
                doc = frame.document;
            }
            return doc;
        }

        // Rails CSRF hack (thanks to Yvan Barthelemy)
        var csrf_token = $('meta[name=csrf-token]').attr('content');
        var csrf_param = $('meta[name=csrf-param]').attr('content');
        if (csrf_param && csrf_token) {
            s.extraData = s.extraData || {};
            s.extraData[csrf_param] = csrf_token;
        }

        // take a breath so that pending repaints get some cpu time before the upload starts
        function doSubmit() {
            // make sure form attrs are set
            var t = $form.attr2('target'), 
                a = $form.attr2('action'), 
                mp = 'multipart/form-data',
                et = $form.attr('enctype') || $form.attr('encoding') || mp;

            // update form attrs in IE friendly way
            form.setAttribute('target',id);
            if (!method || /post/i.test(method) ) {
                form.setAttribute('method', 'POST');
            }
            if (a != s.url) {
                form.setAttribute('action', s.url);
            }

            // ie borks in some cases when setting encoding
            if (! s.skipEncodingOverride && (!method || /post/i.test(method))) {
                $form.attr({
                    encoding: 'multipart/form-data',
                    enctype:  'multipart/form-data'
                });
            }

            // support timout
            if (s.timeout) {
                timeoutHandle = setTimeout(function() { timedOut = true; cb(CLIENT_TIMEOUT_ABORT); }, s.timeout);
            }

            // look for server aborts
            function checkState() {
                try {
                    var state = getDoc(io).readyState;
                    log('state = ' + state);
                    if (state && state.toLowerCase() == 'uninitialized') {
                        setTimeout(checkState,50);
                    }
                }
                catch(e) {
                    log('Server abort: ' , e, ' (', e.name, ')');
                    cb(SERVER_ABORT);
                    if (timeoutHandle) {
                        clearTimeout(timeoutHandle);
                    }
                    timeoutHandle = undefined;
                }
            }

            // add "extra" data to form if provided in options
            var extraInputs = [];
            try {
                if (s.extraData) {
                    for (var n in s.extraData) {
                        if (s.extraData.hasOwnProperty(n)) {
                           // if using the $.param format that allows for multiple values with the same name
                           if($.isPlainObject(s.extraData[n]) && s.extraData[n].hasOwnProperty('name') && s.extraData[n].hasOwnProperty('value')) {
                               extraInputs.push(
                               $('<input type="hidden" name="'+s.extraData[n].name+'">').val(s.extraData[n].value)
                                   .appendTo(form)[0]);
                           } else {
                               extraInputs.push(
                               $('<input type="hidden" name="'+n+'">').val(s.extraData[n])
                                   .appendTo(form)[0]);
                           }
                        }
                    }
                }

                if (!s.iframeTarget) {
                    // add iframe to doc and submit the form
                    $io.appendTo('body');
                }
                if (io.attachEvent) {
                    io.attachEvent('onload', cb);
                }
                else {
                    io.addEventListener('load', cb, false);
                }
                setTimeout(checkState,15);

                try {
                    form.submit();
                } catch(err) {
                    // just in case form has element with name/id of 'submit'
                    var submitFn = document.createElement('form').submit;
                    submitFn.apply(form);
                }
            }
            finally {
                // reset attrs and remove "extra" input elements
                form.setAttribute('action',a);
                form.setAttribute('enctype', et); // #380
                if(t) {
                    form.setAttribute('target', t);
                } else {
                    $form.removeAttr('target');
                }
                $(extraInputs).remove();
            }
        }

        if (s.forceSync) {
            doSubmit();
        }
        else {
            setTimeout(doSubmit, 10); // this lets dom updates render
        }

        var data, doc, domCheckCount = 50, callbackProcessed;

        function cb(e) {
            if (xhr.aborted || callbackProcessed) {
                return;
            }
            
            doc = getDoc(io);
            if(!doc) {
                log('cannot access response document');
                e = SERVER_ABORT;
            }
            if (e === CLIENT_TIMEOUT_ABORT && xhr) {
                xhr.abort('timeout');
                deferred.reject(xhr, 'timeout');
                return;
            }
            else if (e == SERVER_ABORT && xhr) {
                xhr.abort('server abort');
                deferred.reject(xhr, 'error', 'server abort');
                return;
            }

            if (!doc || doc.location.href == s.iframeSrc) {
                // response not received yet
                if (!timedOut) {
                    return;
                }
            }
            if (io.detachEvent) {
                io.detachEvent('onload', cb);
            }
            else {
                io.removeEventListener('load', cb, false);
            }

            var status = 'success', errMsg;
            try {
                if (timedOut) {
                    throw 'timeout';
                }

                var isXml = s.dataType == 'xml' || doc.XMLDocument || $.isXMLDoc(doc);
                log('isXml='+isXml);
                if (!isXml && window.opera && (doc.body === null || !doc.body.innerHTML)) {
                    if (--domCheckCount) {
                        // in some browsers (Opera) the iframe DOM is not always traversable when
                        // the onload callback fires, so we loop a bit to accommodate
                        log('requeing onLoad callback, DOM not available');
                        setTimeout(cb, 250);
                        return;
                    }
                    // let this fall through because server response could be an empty document
                    //log('Could not access iframe DOM after mutiple tries.');
                    //throw 'DOMException: not available';
                }

                //log('response detected');
                var docRoot = doc.body ? doc.body : doc.documentElement;
                xhr.responseText = docRoot ? docRoot.innerHTML : null;
                xhr.responseXML = doc.XMLDocument ? doc.XMLDocument : doc;
                if (isXml) {
                    s.dataType = 'xml';
                }
                xhr.getResponseHeader = function(header){
                    var headers = {'content-type': s.dataType};
                    return headers[header.toLowerCase()];
                };
                // support for XHR 'status' & 'statusText' emulation :
                if (docRoot) {
                    xhr.status = Number( docRoot.getAttribute('status') ) || xhr.status;
                    xhr.statusText = docRoot.getAttribute('statusText') || xhr.statusText;
                }

                var dt = (s.dataType || '').toLowerCase();
                var scr = /(json|script|text)/.test(dt);
                if (scr || s.textarea) {
                    // see if user embedded response in textarea
                    var ta = doc.getElementsByTagName('textarea')[0];
                    if (ta) {
                        xhr.responseText = ta.value;
                        // support for XHR 'status' & 'statusText' emulation :
                        xhr.status = Number( ta.getAttribute('status') ) || xhr.status;
                        xhr.statusText = ta.getAttribute('statusText') || xhr.statusText;
                    }
                    else if (scr) {
                        // account for browsers injecting pre around json response
                        var pre = doc.getElementsByTagName('pre')[0];
                        var b = doc.getElementsByTagName('body')[0];
                        if (pre) {
                            xhr.responseText = pre.textContent ? pre.textContent : pre.innerText;
                        }
                        else if (b) {
                            xhr.responseText = b.textContent ? b.textContent : b.innerText;
                        }
                    }
                }
                else if (dt == 'xml' && !xhr.responseXML && xhr.responseText) {
                    xhr.responseXML = toXml(xhr.responseText);
                }

                try {
                    data = httpData(xhr, dt, s);
                }
                catch (err) {
                    status = 'parsererror';
                    xhr.error = errMsg = (err || status);
                }
            }
            catch (err) {
                log('error caught: ',err);
                status = 'error';
                xhr.error = errMsg = (err || status);
            }

            if (xhr.aborted) {
                log('upload aborted');
                status = null;
            }

            if (xhr.status) { // we've set xhr.status
                status = (xhr.status >= 200 && xhr.status < 300 || xhr.status === 304) ? 'success' : 'error';
            }

            // ordering of these callbacks/triggers is odd, but that's how $.ajax does it
            if (status === 'success') {
                if (s.success) {
                    s.success.call(s.context, data, 'success', xhr);
                }
                deferred.resolve(xhr.responseText, 'success', xhr);
                if (g) {
                    $.event.trigger("ajaxSuccess", [xhr, s]);
                }
            }
            else if (status) {
                if (errMsg === undefined) {
                    errMsg = xhr.statusText;
                }
                if (s.error) {
                    s.error.call(s.context, xhr, status, errMsg);
                }
                deferred.reject(xhr, 'error', errMsg);
                if (g) {
                    $.event.trigger("ajaxError", [xhr, s, errMsg]);
                }
            }

            if (g) {
                $.event.trigger("ajaxComplete", [xhr, s]);
            }

            if (g && ! --$.active) {
                $.event.trigger("ajaxStop");
            }

            if (s.complete) {
                s.complete.call(s.context, xhr, status);
            }

            callbackProcessed = true;
            if (s.timeout) {
                clearTimeout(timeoutHandle);
            }

            // clean up
            setTimeout(function() {
                if (!s.iframeTarget) {
                    $io.remove();
                }
                else { //adding else to clean up existing iframe response.
                    $io.attr('src', s.iframeSrc);
                }
                xhr.responseXML = null;
            }, 100);
        }

        var toXml = $.parseXML || function(s, doc) { // use parseXML if available (jQuery 1.5+)
            if (window.ActiveXObject) {
                doc = new ActiveXObject('Microsoft.XMLDOM');
                doc.async = 'false';
                doc.loadXML(s);
            }
            else {
                doc = (new DOMParser()).parseFromString(s, 'text/xml');
            }
            return (doc && doc.documentElement && doc.documentElement.nodeName != 'parsererror') ? doc : null;
        };
        var parseJSON = $.parseJSON || function(s) {
            /*jslint evil:true */
            return window['eval']('(' + s + ')');
        };

        var httpData = function( xhr, type, s ) { // mostly lifted from jq1.4.4

            var ct = xhr.getResponseHeader('content-type') || '',
                xml = type === 'xml' || !type && ct.indexOf('xml') >= 0,
                data = xml ? xhr.responseXML : xhr.responseText;

            if (xml && data.documentElement.nodeName === 'parsererror') {
                if ($.error) {
                    $.error('parsererror');
                }
            }
            if (s && s.dataFilter) {
                data = s.dataFilter(data, type);
            }
            if (typeof data === 'string') {
                if (type === 'json' || !type && ct.indexOf('json') >= 0) {
                    data = parseJSON(data);
                } else if (type === "script" || !type && ct.indexOf("javascript") >= 0) {
                    $.globalEval(data);
                }
            }
            return data;
        };

        return deferred;
    }
};

/**
 * ajaxForm() provides a mechanism for fully automating form submission.
 *
 * The advantages of using this method instead of ajaxSubmit() are:
 *
 * 1: This method will include coordinates for <input type="image" /> elements (if the element
 *    is used to submit the form).
 * 2. This method will include the submit element's name/value data (for the element that was
 *    used to submit the form).
 * 3. This method binds the submit() method to the form for you.
 *
 * The options argument for ajaxForm works exactly as it does for ajaxSubmit.  ajaxForm merely
 * passes the options argument along after properly binding events for submit elements and
 * the form itself.
 */
$.fn.ajaxForm = function(options) {
    options = options || {};
    options.delegation = options.delegation && $.isFunction($.fn.on);

    // in jQuery 1.3+ we can fix mistakes with the ready state
    if (!options.delegation && this.length === 0) {
        var o = { s: this.selector, c: this.context };
        if (!$.isReady && o.s) {
            log('DOM not ready, queuing ajaxForm');
            $(function() {
                $(o.s,o.c).ajaxForm(options);
            });
            return this;
        }
        // is your DOM ready?  http://docs.jquery.com/Tutorials:Introducing_$(document).ready()
        log('terminating; zero elements found by selector' + ($.isReady ? '' : ' (DOM not ready)'));
        return this;
    }

    if ( options.delegation ) {
        $(document)
            .off('submit.form-plugin', this.selector, doAjaxSubmit)
            .off('click.form-plugin', this.selector, captureSubmittingElement)
            .on('submit.form-plugin', this.selector, options, doAjaxSubmit)
            .on('click.form-plugin', this.selector, options, captureSubmittingElement);
        return this;
    }

    return this.ajaxFormUnbind()
        .bind('submit.form-plugin', options, doAjaxSubmit)
        .bind('click.form-plugin', options, captureSubmittingElement);
};

// private event handlers
function doAjaxSubmit(e) {
    /*jshint validthis:true */
    var options = e.data;
    if (!e.isDefaultPrevented()) { // if event has been canceled, don't proceed
        e.preventDefault();
        $(e.target).ajaxSubmit(options); // #365
    }
}

function captureSubmittingElement(e) {
    /*jshint validthis:true */
    var target = e.target;
    var $el = $(target);
    if (!($el.is("[type=submit],[type=image]"))) {
        // is this a child element of the submit el?  (ex: a span within a button)
        var t = $el.closest('[type=submit]');
        if (t.length === 0) {
            return;
        }
        target = t[0];
    }
    var form = this;
    form.clk = target;
    if (target.type == 'image') {
        if (e.offsetX !== undefined) {
            form.clk_x = e.offsetX;
            form.clk_y = e.offsetY;
        } else if (typeof $.fn.offset == 'function') {
            var offset = $el.offset();
            form.clk_x = e.pageX - offset.left;
            form.clk_y = e.pageY - offset.top;
        } else {
            form.clk_x = e.pageX - target.offsetLeft;
            form.clk_y = e.pageY - target.offsetTop;
        }
    }
    // clear form vars
    setTimeout(function() { form.clk = form.clk_x = form.clk_y = null; }, 100);
}


// ajaxFormUnbind unbinds the event handlers that were bound by ajaxForm
$.fn.ajaxFormUnbind = function() {
    return this.unbind('submit.form-plugin click.form-plugin');
};

/**
 * formToArray() gathers form element data into an array of objects that can
 * be passed to any of the following ajax functions: $.get, $.post, or load.
 * Each object in the array has both a 'name' and 'value' property.  An example of
 * an array for a simple login form might be:
 *
 * [ { name: 'username', value: 'jresig' }, { name: 'password', value: 'secret' } ]
 *
 * It is this array that is passed to pre-submit callback functions provided to the
 * ajaxSubmit() and ajaxForm() methods.
 */
$.fn.formToArray = function(semantic, elements) {
    var a = [];
    if (this.length === 0) {
        return a;
    }

    var form = this[0];
    var formId = this.attr('id');
    var els = semantic ? form.getElementsByTagName('*') : form.elements;
    var els2;

    if (els && !/MSIE [678]/.test(navigator.userAgent)) { // #390
        els = $(els).get();  // convert to standard array
    }

    // #386; account for inputs outside the form which use the 'form' attribute
    if ( formId ) {
        els2 = $(':input[form="' + formId + '"]').get(); // hat tip @thet
        if ( els2.length ) {
            els = (els || []).concat(els2);
        }
    }

    if (!els || !els.length) {
        return a;
    }

    var i,j,n,v,el,max,jmax;
    for(i=0, max=els.length; i < max; i++) {
        el = els[i];
        n = el.name;
        if (!n || el.disabled) {
            continue;
        }

        if (semantic && form.clk && el.type == "image") {
            // handle image inputs on the fly when semantic == true
            if(form.clk == el) {
                a.push({name: n, value: $(el).val(), type: el.type });
                a.push({name: n+'.x', value: form.clk_x}, {name: n+'.y', value: form.clk_y});
            }
            continue;
        }

        v = $.fieldValue(el, true);
        if (v && v.constructor == Array) {
            if (elements) {
                elements.push(el);
            }
            for(j=0, jmax=v.length; j < jmax; j++) {
                a.push({name: n, value: v[j]});
            }
        }
        else if (feature.fileapi && el.type == 'file') {
            if (elements) {
                elements.push(el);
            }
            var files = el.files;
            if (files.length) {
                for (j=0; j < files.length; j++) {
                    a.push({name: n, value: files[j], type: el.type});
                }
            }
            else {
                // #180
                a.push({ name: n, value: '', type: el.type });
            }
        }
        else if (v !== null && typeof v != 'undefined') {
            if (elements) {
                elements.push(el);
            }
            a.push({name: n, value: v, type: el.type, required: el.required});
        }
    }

    if (!semantic && form.clk) {
        // input type=='image' are not found in elements array! handle it here
        var $input = $(form.clk), input = $input[0];
        n = input.name;
        if (n && !input.disabled && input.type == 'image') {
            a.push({name: n, value: $input.val()});
            a.push({name: n+'.x', value: form.clk_x}, {name: n+'.y', value: form.clk_y});
        }
    }
    return a;
};

/**
 * Serializes form data into a 'submittable' string. This method will return a string
 * in the format: name1=value1&amp;name2=value2
 */
$.fn.formSerialize = function(semantic) {
    //hand off to jQuery.param for proper encoding
    return $.param(this.formToArray(semantic));
};

/**
 * Serializes all field elements in the jQuery object into a query string.
 * This method will return a string in the format: name1=value1&amp;name2=value2
 */
$.fn.fieldSerialize = function(successful) {
    var a = [];
    this.each(function() {
        var n = this.name;
        if (!n) {
            return;
        }
        var v = $.fieldValue(this, successful);
        if (v && v.constructor == Array) {
            for (var i=0,max=v.length; i < max; i++) {
                a.push({name: n, value: v[i]});
            }
        }
        else if (v !== null && typeof v != 'undefined') {
            a.push({name: this.name, value: v});
        }
    });
    //hand off to jQuery.param for proper encoding
    return $.param(a);
};

/**
 * Returns the value(s) of the element in the matched set.  For example, consider the following form:
 *
 *  <form><fieldset>
 *      <input name="A" type="text" />
 *      <input name="A" type="text" />
 *      <input name="B" type="checkbox" value="B1" />
 *      <input name="B" type="checkbox" value="B2"/>
 *      <input name="C" type="radio" value="C1" />
 *      <input name="C" type="radio" value="C2" />
 *  </fieldset></form>
 *
 *  var v = $('input[type=text]').fieldValue();
 *  // if no values are entered into the text inputs
 *  v == ['','']
 *  // if values entered into the text inputs are 'foo' and 'bar'
 *  v == ['foo','bar']
 *
 *  var v = $('input[type=checkbox]').fieldValue();
 *  // if neither checkbox is checked
 *  v === undefined
 *  // if both checkboxes are checked
 *  v == ['B1', 'B2']
 *
 *  var v = $('input[type=radio]').fieldValue();
 *  // if neither radio is checked
 *  v === undefined
 *  // if first radio is checked
 *  v == ['C1']
 *
 * The successful argument controls whether or not the field element must be 'successful'
 * (per http://www.w3.org/TR/html4/interact/forms.html#successful-controls).
 * The default value of the successful argument is true.  If this value is false the value(s)
 * for each element is returned.
 *
 * Note: This method *always* returns an array.  If no valid value can be determined the
 *    array will be empty, otherwise it will contain one or more values.
 */
$.fn.fieldValue = function(successful) {
    for (var val=[], i=0, max=this.length; i < max; i++) {
        var el = this[i];
        var v = $.fieldValue(el, successful);
        if (v === null || typeof v == 'undefined' || (v.constructor == Array && !v.length)) {
            continue;
        }
        if (v.constructor == Array) {
            $.merge(val, v);
        }
        else {
            val.push(v);
        }
    }
    return val;
};

/**
 * Returns the value of the field element.
 */
$.fieldValue = function(el, successful) {
    var n = el.name, t = el.type, tag = el.tagName.toLowerCase();
    if (successful === undefined) {
        successful = true;
    }

    if (successful && (!n || el.disabled || t == 'reset' || t == 'button' ||
        (t == 'checkbox' || t == 'radio') && !el.checked ||
        (t == 'submit' || t == 'image') && el.form && el.form.clk != el ||
        tag == 'select' && el.selectedIndex == -1)) {
            return null;
    }

    if (tag == 'select') {
        var index = el.selectedIndex;
        if (index < 0) {
            return null;
        }
        var a = [], ops = el.options;
        var one = (t == 'select-one');
        var max = (one ? index+1 : ops.length);
        for(var i=(one ? index : 0); i < max; i++) {
            var op = ops[i];
            if (op.selected) {
                var v = op.value;
                if (!v) { // extra pain for IE...
                    v = (op.attributes && op.attributes.value && !(op.attributes.value.specified)) ? op.text : op.value;
                }
                if (one) {
                    return v;
                }
                a.push(v);
            }
        }
        return a;
    }
    return $(el).val();
};

/**
 * Clears the form data.  Takes the following actions on the form's input fields:
 *  - input text fields will have their 'value' property set to the empty string
 *  - select elements will have their 'selectedIndex' property set to -1
 *  - checkbox and radio inputs will have their 'checked' property set to false
 *  - inputs of type submit, button, reset, and hidden will *not* be effected
 *  - button elements will *not* be effected
 */
$.fn.clearForm = function(includeHidden) {
    return this.each(function() {
        $('input,select,textarea', this).clearFields(includeHidden);
    });
};

/**
 * Clears the selected form elements.
 */
$.fn.clearFields = $.fn.clearInputs = function(includeHidden) {
    var re = /^(?:color|date|datetime|email|month|number|password|range|search|tel|text|time|url|week)$/i; // 'hidden' is not in this list
    return this.each(function() {
        var t = this.type, tag = this.tagName.toLowerCase();
        if (re.test(t) || tag == 'textarea') {
            this.value = '';
        }
        else if (t == 'checkbox' || t == 'radio') {
            this.checked = false;
        }
        else if (tag == 'select') {
            this.selectedIndex = -1;
        }
        else if (t == "file") {
            if (/MSIE/.test(navigator.userAgent)) {
                $(this).replaceWith($(this).clone(true));
            } else {
                $(this).val('');
            }
        }
        else if (includeHidden) {
            // includeHidden can be the value true, or it can be a selector string
            // indicating a special test; for example:
            //  $('#myForm').clearForm('.special:hidden')
            // the above would clean hidden inputs that have the class of 'special'
            if ( (includeHidden === true && /hidden/.test(t)) ||
                 (typeof includeHidden == 'string' && $(this).is(includeHidden)) ) {
                this.value = '';
            }
        }
    });
};

/**
 * Resets the form data.  Causes all form elements to be reset to their original value.
 */
$.fn.resetForm = function() {
    return this.each(function() {
        // guard against an input with the name of 'reset'
        // note that IE reports the reset function as an 'object'
        if (typeof this.reset == 'function' || (typeof this.reset == 'object' && !this.reset.nodeType)) {
            this.reset();
        }
    });
};

/**
 * Enables or disables any matching elements.
 */
$.fn.enable = function(b) {
    if (b === undefined) {
        b = true;
    }
    return this.each(function() {
        this.disabled = !b;
    });
};

/**
 * Checks/unchecks any matching checkboxes or radio buttons and
 * selects/deselects and matching option elements.
 */
$.fn.selected = function(select) {
    if (select === undefined) {
        select = true;
    }
    return this.each(function() {
        var t = this.type;
        if (t == 'checkbox' || t == 'radio') {
            this.checked = select;
        }
        else if (this.tagName.toLowerCase() == 'option') {
            var $sel = $(this).parent('select');
            if (select && $sel[0] && $sel[0].type == 'select-one') {
                // deselect all other options
                $sel.find('option').selected(false);
            }
            this.selected = select;
        }
    });
};

// expose debug var
$.fn.ajaxSubmit.debug = false;

// helper fn for console logging
function log() {
    if (!$.fn.ajaxSubmit.debug) {
        return;
    }
    var msg = '[jquery.form] ' + Array.prototype.join.call(arguments,'');
    if (window.console && window.console.log) {
        window.console.log(msg);
    }
    else if (window.opera && window.opera.postError) {
        window.opera.postError(msg);
    }
}

}));


/* - ++resource++plone.app.jquerytools.overlayhelpers.js - */
// https://www.gov.br/portal_javascripts/++resource++plone.app.jquerytools.overlayhelpers.js?original=1
var pb={spinner:{},overlay_counter:1};jQuery.tools.overlay.conf.oneInstance=false;(function($){jQuery.tools.overlay.addEffect('default',
function(pos,onLoad){var conf=this.getConf(),w=$(window),ovl=this.getOverlay(),op=ovl.parent().offsetParent().offset();if(!conf.fixed){pos.top+=w.scrollTop()-op.top;pos.left+=w.scrollLeft()-op.left}
pos.position=conf.fixed?'fixed':'absolute';ovl.css(pos).fadeIn(conf.speed,onLoad)}, function(onClose){this.getOverlay().fadeOut(this.getConf().closeSpeed,onClose)});pb.spinner.show=function(){$('body').css('cursor','wait')};pb.spinner.hide=function(){$('body').css('cursor','')}}(jQuery));jQuery(function($){$.fn.prepOverlay=function(pba){return this.each(function(){var o,pbo,config,onBeforeLoad,onLoad,src,parts;o=$(this);pbo=$.extend(true,{'width':'60%'},pba);config=pbo.config||{};onBeforeLoad=pb[pbo.subtype];if(onBeforeLoad){config.onBeforeLoad=onBeforeLoad}
onLoad=config.onLoad;config.onLoad=function(){if(onLoad){onLoad.apply(this,arguments)}
pb.fi_focus(this.getOverlay())};src=o.attr('href')||o.attr('src')||o.attr('action');if(pbo.urlmatch){src=src.replace(new RegExp(pbo.urlmatch),pbo.urlreplace)}
if(pbo.subtype==='inline'){src=src.replace(/^.+#/,'#');$("[id='"+src.replace('#','')+"']").addClass('overlay');o.removeAttr('href').attr('rel',src);o.overlay()} else{pbo.nt='pb_'+pb.overlay_counter;pb.overlay_counter+=1;pbo.selector=pbo.filter||pbo.selector;if(!pbo.selector){parts=src.split(' ');src=parts.shift();pbo.selector=parts.join(' ')}
pbo.src=src;pbo.config=config;pbo.source=o;pb.remove_overlay(o);o.data('pbo',pbo);o.attr('rel','#'+pbo.nt);o.addClass('link-overlay');register_handler=function(evt,o,hnd){switch(evt){case undefined:o.click(hnd);break;case 'click':o.click(hnd);break;case 'hover':o.hover(hnd);break;case 'dblclick':o.dblclick(hnd);break;default:throw "Unsupported event"}}
switch(pbo.subtype){case 'image':register_handler(pbo.event,o,pb.image_click);break;case 'ajax':register_handler(pbo.event,o,pb.ajax_click);break;case 'iframe':pb.create_content_div(pbo);o.overlay(config);break;default:throw "Unsupported overlay type"}
o.css('cursor','pointer')}})};pb.remove_overlay=function(o){var old_data=o.data('pbo');if(old_data){switch(old_data.subtype){case 'image':o.unbind('click',pb.image_click);break;case 'ajax':o.unbind('click',pb.ajax_click);break;default:o.unbind('click');break}
if(old_data.nt){$('#'+old_data.nt).remove()}}};pb.create_content_div=function(pbo,trigger){var content,close_message,pbw=pbo.width;if(typeof close_box_message==='undefined'){close_message='Close this box.'} else{close_message=close_box_message}
content=$('<div id="'+pbo.nt+'" class="overlay overlay-'+pbo.subtype+' '+(pbo.cssclass||'')+'"><div class="close"><a href="#" class="hiddenStructure" title="Close this box">'+close_message+'</a></div></div>');content.data('pbo',pbo);if(pbw){if(pbw.indexOf('%')>0){content.width(parseInt(pbw,10)/100 * $(window).width())} else{content.width(pbw)}}
content.appendTo($("body"));return content};pb.image_click=function(event){var ethis,content,api,img,el,pbo;ethis=$(this);pbo=ethis.data('pbo');content=$(ethis.attr('rel'));if(!content.length){content=pb.create_content_div(pbo);content.overlay(pbo.config)}
api=content.overlay();if(content.find('img').length===0){if(pbo.src){pb.spinner.show();img=new Image();img.src=pbo.src;el=$(img);content.append(el.addClass('pb-image'));el.load(function(){pb.spinner.hide();api.load()})}} else{api.load()}
return false};pb.fi_focus=function(jqo){if(!jqo.find("form div.error :input:first").focus().length){jqo.find("form :input:visible:first").focus()}};pb.ajax_error_recover=function(responseText,selector){var tcontent=$('<div/>').append(responseText.replace(/<script(.|\s)*?\/script>/gi,""));return selector?tcontent.find(selector):tcontent};pb.add_ajax_load=function(form){if(form.find('input[name=ajax_load]').length===0){form.prepend($('<input type="hidden" name="ajax_load" value="'+(new Date().getTime())+'" />'))}};pb.prep_ajax_form=function(form){var ajax_parent=form.closest('.pb-ajax'),data_parent=ajax_parent.closest('.overlay-ajax'),pbo=data_parent.data('pbo'),formtarget=pbo.formselector,closeselector=pbo.closeselector,beforepost=pbo.beforepost,afterpost=pbo.afterpost,noform=pbo.noform,api=data_parent.overlay(),selector=pbo.selector,options={};options.beforeSerialize=function(){pb.spinner.show()};if(beforepost){options.beforeSubmit=function(arr,form,options){return beforepost(form,arr,options)}}
options.success=function(responseText,statusText,xhr,form){$(document).trigger('formOverlayStart',[this,responseText,statusText,xhr,form]);var el,myform,success,target,scripts=[],filteredResponseText;success=statusText==="success"||statusText==="notmodified";if(!success){responseText=responseText.responseText}
filteredResponseText=responseText.replace(/<script(.|\s)*?\/script>/gi,"");el=$('<div />').append(selector?$('<div />').append(filteredResponseText).find(selector):filteredResponseText);if(success&&afterpost){afterpost(el,data_parent)}
myform=el.find(formtarget);if(success&&myform.length){ajax_parent.empty().append(el);try{$.buildFragment([responseText],[document],scripts)} catch(e){$.buildFragment([responseText],document,scripts)}
if(scripts.length){$.each(scripts, function(){$.globalEval(this.text||this.textContent||this.innerHTML||"")})}
if($.fn.ploneTabInit){el.ploneTabInit()}
pb.fi_focus(ajax_parent);pb.add_ajax_load(myform);myform.ajaxForm(options);if(closeselector){el.find(closeselector).click(function(event){api.close();return false})}
$(document).trigger('formOverlayLoadSuccess',[this,myform,api,pb,ajax_parent])} else{if(success){if(typeof noform==="function"){noform=noform(el,pbo)}} else{noform=statusText}
switch(noform){case 'close':api.close();break;case 'reload':api.close();pb.spinner.show();location.replace(location.href);break;case 'redirect':api.close();pb.spinner.show();target=pbo.redirect;if(typeof target==="function"){target=target(el,responseText,pbo)}
location.replace(target);break;default:if(el.children()){ajax_parent.empty().append(el)} else{api.close()}
break}
$(document).trigger('formOverlayLoadFailure',[this,myform,api,pb,ajax_parent,noform])}
pb.spinner.hide()};options.error=options.success;pb.add_ajax_load(form);form.ajaxForm(options)};pb.ajax_click=function(event){var ethis=$(this),pbo,content,api,src,el,selector,formtarget,closeselector,sep,scripts=[],e;e=$.Event();e.type="beforeAjaxClickHandled";$(document).trigger(e,[this,event]);if(e.isDefaultPrevented()){return false}
pbo=ethis.data('pbo');content=pb.create_content_div(pbo,ethis);content.overlay(pbo.config,ethis);api=content.overlay();src=pbo.src;selector=pbo.selector;formtarget=pbo.formselector;closeselector=pbo.closeselector;pb.spinner.show();$(this).find("input.submitting").removeClass('submitting');el=$('div.pb-ajax',content);if(el.length===0){el=$('<div class="pb-ajax" />');content.append(el)}
if(api.getConf().fixed){el.css('max-height',Math.floor($(window).height() * 0.75))}
sep=(src.indexOf('?')>=0)?'&':'?';src+=sep+"ajax_load="+(new Date().getTime());if(selector){src+=' '+selector}
el[0].handle_load_inside_overlay=function(responseText,errorText){var ele,target;ele=$(this);if(errorText==='error'){ele.append(pb.ajax_error_recover(responseText,selector))} else if(!responseText.length){ele.append(ajax_noresponse_message||'No response from server.')}
ele.wrapInner('<div />');if(formtarget){target=ele.find(formtarget);if(target.length>0){pb.prep_ajax_form(target)}}
if(closeselector){ele.find(closeselector).click(function(event){api.close();return false})}
try{$.buildFragment([responseText],[document],scripts)} catch(e){$.buildFragment([responseText],document,scripts)}
if(scripts.length){$.each(scripts, function(){$.globalEval(this.text||this.textContent||this.innerHTML||"")})}
if($.fn.ploneTabInit){ele.ploneTabInit()}
api.onClose=function(){content.remove()};$(document).trigger('loadInsideOverlay',[this,responseText,errorText,api])};var data=null;var form=$(pbo.source.context);var method=form.attr('method');if(method&&method.toLowerCase()=='post'){data=form.serializeArray()}
el.load(src,data, function(responseText,errorText){el[0].handle_load_inside_overlay.apply(this,[responseText,errorText]);pb.spinner.hide();api.load();return true});return false};pb.iframe=function(){var content,pbo;pb.spinner.show();content=this.getOverlay();pbo=this.getTrigger().data('pbo');if(content.find('iframe').length===0&&pbo.src){content.append('<iframe src="'+pbo.src+'" width="'+content.width()+'" height="'+content.height()+'" onload="pb.spinner.hide()"/>')} else{pb.spinner.hide()}
return true}});

/* - ++resource++plone.app.jquerytools.dateinput.js - */
!function($,undefined){function dayAm(year,month){return new Date(year,month+1,0).getDate()}function zeropad(val,len){for(val=""+val,len=len||2;val.length<len;)val="0"+val;return val}function format(formatter,date,text,lang){var d=date.getDate(),D=date.getDay(),m=date.getMonth(),y=date.getFullYear(),flags={d:d,dd:zeropad(d),ddd:LABELS[lang].shortDays[D],dddd:LABELS[lang].days[D],m:m+1,mm:zeropad(m+1),mmm:LABELS[lang].shortMonths[m],mmmm:LABELS[lang].months[m],yy:String(y).slice(2),yyyy:y},ret=formatters[formatter](text,date,flags,lang);return tmpTag.html(ret).html()}function integer(val){return parseInt(val,10)}function isSameDay(d1,d2){return d1.getFullYear()===d2.getFullYear()&&d1.getMonth()==d2.getMonth()&&d1.getDate()==d2.getDate()}function parseDate(val){if(val!==undefined){if(val.constructor==Date)return val;if("string"==typeof val){var els=val.split("-");if(3==els.length)return new Date(integer(els[0]),integer(els[1])-1,integer(els[2]));if(!/^-?\d+$/.test(val))return;val=integer(val)}var date=new Date;return date.setDate(date.getDate()+val),date}}function Dateinput(input,conf){function select(date,conf,e){return input.attr("readonly")?void self.hide(e):(value=date,currYear=date.getFullYear(),currMonth=date.getMonth(),currDay=date.getDate(),e||(e=$.Event("api")),"click"!=e.type||/msie/.test(navigator.userAgent.toLowerCase())||input.focus(),e.type="beforeChange",fire.trigger(e,[date]),void(e.isDefaultPrevented()||(input.val(format(conf.formatter,date,conf.format,conf.lang)),e.type="change",e.target=input[0],fire.trigger(e),input.data("date",date),self.hide(e))))}function onShow(ev){ev.type="onShow",fire.trigger(ev),$(document).on("keydown.d",function(e){if(e.ctrlKey)return!0;var key=e.keyCode;if(8==key||46==key)return input.val(""),self.hide(e);if(27==key||9==key)return self.hide(e);if($(KEYS).index(key)>=0){if(!opened)return self.show(e),e.preventDefault();var days=$("#"+css.weeks+" a"),el=$("."+css.focus),index=days.index(el);return el.removeClass(css.focus),74==key||40==key?index+=7:75==key||38==key?index-=7:76==key||39==key?index+=1:72!=key&&37!=key||(index-=1),index>41?(self.addMonth(),el=$("#"+css.weeks+" a:eq("+(index-42)+")")):index<0?(self.addMonth(-1),el=$("#"+css.weeks+" a:eq("+(index+42)+")")):el=days.eq(index),el.addClass(css.focus),e.preventDefault()}return 34==key?self.addMonth():33==key?self.addMonth(-1):36==key?self.today():(13==key&&($(e.target).is("select")||$("."+css.focus).click()),$([16,17,18,9]).index(key)>=0)}),$(document).on("click.d",function(e){var el=e.target;el.id==css.root||$(el).parents("#"+css.root).length||el==input[0]||trigger&&el==trigger[0]||self.hide(e)})}var trigger,pm,nm,currYear,currMonth,currDay,opened,original,self=this,now=new Date,yearNow=now.getFullYear(),css=conf.css,labels=LABELS[conf.lang],root=$("#"+css.root),title=root.find("#"+css.title),value=input.attr("data-value")||conf.value||input.val(),min=input.attr("min")||conf.min,max=input.attr("max")||conf.max;if(0===min&&(min="0"),value=parseDate(value)||now,min=parseDate(min||new Date(yearNow+conf.yearRange[0],0,1)),max=parseDate(max||new Date(yearNow+conf.yearRange[1]+1,0,0)),!labels)throw"Dateinput: invalid language: "+conf.lang;if("date"==input.attr("type")){var original=input.clone(),def=original.wrap("<div/>").parent().html(),clone=$(def.replace(/type/i,"type=text data-orig-type"));conf.value&&clone.val(conf.value),input.replaceWith(clone),input=clone}input.addClass(css.input);var fire=input.add(self);if(!root.length){if(root=$("<div><div><a/><div/><a/></div><div><div/><div/></div></div>").hide().css({position:"absolute"}).attr("id",css.root),root.children().eq(0).attr("id",css.head).end().eq(1).attr("id",css.body).children().eq(0).attr("id",css.days).end().eq(1).attr("id",css.weeks).end().end().end().find("a").eq(0).attr("id",css.prev).end().eq(1).attr("id",css.next),title=root.find("#"+css.head).find("div").attr("id",css.title),conf.selectors){var monthSelector=$("<select/>").attr("id",css.month),yearSelector=$("<select/>").attr("id",css.year);title.html(monthSelector.add(yearSelector))}for(var days=root.find("#"+css.days),d=0;d<7;d++)days.append($("<span/>").text(labels.shortDays[(d+conf.firstDay)%7]));$("body").append(root)}conf.trigger&&(trigger=$("<a/>").attr("href","#").addClass(css.trigger).click(function(e){return conf.toggle?self.toggle():self.show(),e.preventDefault()}).insertAfter(input));var weeks=root.find("#"+css.weeks);yearSelector=root.find("#"+css.year),monthSelector=root.find("#"+css.month),$.extend(self,{show:function(e){if(!input.attr("disabled")&&!opened&&(e=e||$.Event(),e.type="onBeforeShow",fire.trigger(e),!e.isDefaultPrevented())){$.each(instances,function(){this.hide()}),opened=!0,monthSelector.off("change").change(function(){self.setValue(integer(yearSelector.val()),integer($(this).val()))}),yearSelector.off("change").change(function(){self.setValue(integer($(this).val()),integer(monthSelector.val()))}),pm=root.find("#"+css.prev).off("click").click(function(e){return pm.hasClass(css.disabled)||self.addMonth(-1),!1}),nm=root.find("#"+css.next).off("click").click(function(e){return nm.hasClass(css.disabled)||self.addMonth(),!1}),self.setValue(value);var pos=input.offset();return/iPad/i.test(navigator.userAgent)&&(pos.top-=$(window).scrollTop()),root.css({top:pos.top+input.outerHeight(!0)+conf.offset[0],left:pos.left+conf.offset[1]}),conf.speed?root.show(conf.speed,function(){onShow(e)}):(root.show(),onShow(e)),self}},setValue:function(year,month,day){var date=integer(month)>=-1?new Date(integer(year),integer(month),integer(day==undefined||isNaN(day)?1:day)):year||value;if(date<min?date=min:date>max&&(date=max),"string"==typeof year&&(date=parseDate(year)),year=date.getFullYear(),month=date.getMonth(),day=date.getDate(),month==-1?(month=11,year--):12==month&&(month=0,year++),!opened)return select(date,conf),self;currMonth=month,currYear=year,currDay=day;var week,tmp=new Date(year,month,1-conf.firstDay),begin=tmp.getDay(),days=dayAm(year,month),prevDays=dayAm(year,month-1);if(conf.selectors){monthSelector.empty(),$.each(labels.months,function(i,m){min<new Date(year,i+1,1)&&max>new Date(year,i,0)&&monthSelector.append($("<option/>").html(m).attr("value",i))}),yearSelector.empty();for(var yearNow=now.getFullYear(),i=yearNow+conf.yearRange[0];i<yearNow+conf.yearRange[1];i++)min<new Date(i+1,0,1)&&max>new Date(i,0,0)&&yearSelector.append($("<option/>").text(i));monthSelector.val(month),yearSelector.val(year)}else title.html(labels.months[month]+" "+year);weeks.empty(),pm.add(nm).removeClass(css.disabled);for(var a,num,j=begin?0:-7;j<(begin?42:35);j++)a=$("<a/>"),j%7===0&&(week=$("<div/>").addClass(css.week),weeks.append(week)),j<begin?(a.addClass(css.off),num=prevDays-begin+j+1,date=new Date(year,month-1,num)):j>=begin+days?(a.addClass(css.off),num=j-days-begin+1,date=new Date(year,month+1,num)):(num=j-begin+1,date=new Date(year,month,num),isSameDay(value,date)?a.attr("id",css.current).addClass(css.focus):isSameDay(now,date)&&a.attr("id",css.today)),min&&date<min&&a.add(pm).addClass(css.disabled),max&&date>max&&a.add(nm).addClass(css.disabled),a.attr("href","#"+num).text(num).data("date",date),week.append(a);return weeks.find("a").click(function(e){var el=$(this);return el.hasClass(css.disabled)||($("#"+css.current).removeAttr("id"),el.attr("id",css.current),select(el.data("date"),conf,e)),!1}),css.sunday&&weeks.find("."+css.week).each(function(){var beg=conf.firstDay?7-conf.firstDay:0;$(this).children().slice(beg,beg+1).addClass(css.sunday)}),self},setMin:function(val,fit){return min=parseDate(val),fit&&value<min&&self.setValue(min),self},setMax:function(val,fit){return max=parseDate(val),fit&&value>max&&self.setValue(max),self},today:function(){return self.setValue(now)},addDay:function(amount){return this.setValue(currYear,currMonth,currDay+(amount||1))},addMonth:function(amount){var targetMonth=currMonth+(amount||1),daysInTargetMonth=dayAm(currYear,targetMonth),targetDay=currDay<=daysInTargetMonth?currDay:daysInTargetMonth;return this.setValue(currYear,targetMonth,targetDay)},addYear:function(amount){return this.setValue(currYear+(amount||1),currMonth,currDay)},destroy:function(){input.add(document).off("click.d keydown.d"),root.add(trigger).remove(),input.removeData("dateinput").removeClass(css.input),original&&input.replaceWith(original)},hide:function(e){if(opened){if(e=$.Event(),e.type="onHide",fire.trigger(e),e.isDefaultPrevented())return;$(document).off("click.d keydown.d"),root.hide(),opened=!1}return self},toggle:function(){return self.isOpen()?self.hide():self.show()},getConf:function(){return conf},getInput:function(){return input},getCalendar:function(){return root},getValue:function(dateFormat){return dateFormat?format(conf.formatter,value,dateFormat,conf.lang):value},isOpen:function(){return opened}}),$.each(["onBeforeShow","onShow","change","onHide"],function(i,name){$.isFunction(conf[name])&&$(self).on(name,conf[name]),self[name]=function(fn){return fn&&$(self).on(name,fn),self}}),conf.editable||input.on("focus.d click.d",self.show).keydown(function(e){var key=e.keyCode;return!opened&&$(KEYS).index(key)>=0?(self.show(e),e.preventDefault()):(8!=key&&46!=key||input.val(""),!!(e.shiftKey||e.ctrlKey||e.altKey||9==key)||e.preventDefault())}),parseDate(input.val())&&select(value,conf)}$.tools=$.tools||{version:"@VERSION"};var tool,instances=[],formatters={},KEYS=[75,76,38,39,74,72,40,37],LABELS={};tool=$.tools.dateinput={conf:{format:"mm/dd/yy",formatter:"default",selectors:!1,yearRange:[-5,5],lang:"en",offset:[0,0],speed:0,firstDay:0,min:undefined,max:undefined,trigger:0,toggle:0,editable:0,css:{prefix:"cal",input:"date",root:0,head:0,title:0,prev:0,next:0,month:0,year:0,days:0,body:0,weeks:0,today:0,current:0,week:0,off:0,sunday:0,focus:0,disabled:0,trigger:0}},addFormatter:function(name,fn){formatters[name]=fn},localize:function(language,labels){$.each(labels,function(key,val){labels[key]=val.split(",")}),LABELS[language]=labels}},tool.localize("en",{months:"January,February,March,April,May,June,July,August,September,October,November,December",shortMonths:"Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec",days:"Sunday,Monday,Tuesday,Wednesday,Thursday,Friday,Saturday",shortDays:"Sun,Mon,Tue,Wed,Thu,Fri,Sat"});var tmpTag=$("<a/>");tool.addFormatter("default",function(text,date,flags,lang){return text.replace(/d{1,4}|m{1,4}|yy(?:yy)?|"[^"]*"|'[^']*'/g,function($0){return $0 in flags?flags[$0]:$0})}),tool.addFormatter("prefixed",function(text,date,flags,lang){return text.replace(/%(d{1,4}|m{1,4}|yy(?:yy)?|"[^"]*"|'[^']*')/g,function($0,$1){return $1 in flags?flags[$1]:$0})}),$.expr[":"].date=function(el){var type=el.getAttribute("type");return type&&"date"==type||!!$(el).data("dateinput")},$.fn.dateinput=function(conf){if(this.data("dateinput"))return this;conf=$.extend(!0,{},tool.conf,conf),$.each(conf.css,function(key,val){val||"prefix"==key||(conf.css[key]=(conf.css.prefix||"")+(val||key))});var els;return this.each(function(){var el=new Dateinput($(this),conf);instances.push(el);var input=el.getInput().data("dateinput",el);els=els?els.add(input):input}),els?els:this}}(jQuery);

/* - ++resource++plone.app.jquerytools.tooltip.js - */
!function($){function getPosition(trigger,tip,conf){var top=conf.relative?trigger.position().top:trigger.offset().top,left=conf.relative?trigger.position().left:trigger.offset().left,pos=conf.position[0];top-=tip.outerHeight()-conf.offset[0],left+=trigger.outerWidth()+conf.offset[1],/iPad/i.test(navigator.userAgent)&&(top-=$(window).scrollTop());var height=tip.outerHeight()+trigger.outerHeight();"center"==pos&&(top+=height/2),"bottom"==pos&&(top+=height),pos=conf.position[1];var width=tip.outerWidth()+trigger.outerWidth();return"center"==pos&&(left-=width/2),"left"==pos&&(left-=width),{top:top,left:left}}function Tooltip(trigger,conf){var tip,shown,self=this,fire=trigger.add(self),timer=0,pretimer=0,title=trigger.attr("title"),tipAttr=trigger.attr("data-tooltip"),effect=effects[conf.effect],isInput=trigger.is(":input"),isWidget=isInput&&trigger.is(":checkbox, :radio, select, :button, :submit"),type=trigger.attr("type"),evt=conf.events[type]||conf.events[isInput?isWidget?"widget":"input":"def"];if(!effect)throw'Nonexistent effect "'+conf.effect+'"';if(evt=evt.split(/,\s*/),2!=evt.length)throw"Tooltip: bad events configuration for "+type;trigger.on(evt[0],function(e){clearTimeout(timer),conf.predelay?pretimer=setTimeout(function(){self.show(e)},conf.predelay):self.show(e)}).on(evt[1],function(e){clearTimeout(pretimer),conf.delay?timer=setTimeout(function(){self.hide(e)},conf.delay):self.hide(e)}),title&&conf.cancelDefault&&(trigger.removeAttr("title"),trigger.data("title",title)),$.extend(self,{show:function(e){if(!tip&&(tipAttr?tip=$(tipAttr):conf.tip?tip=$(conf.tip).eq(0):title?tip=$(conf.layout).addClass(conf.tipClass).appendTo(document.body).hide().append(title):(tip=trigger.find("."+conf.tipClass),tip.length||(tip=trigger.next()),tip.length||(tip=trigger.parent().next())),!tip.length))throw"Cannot find tooltip for "+trigger;if(self.isShown())return self;tip.stop(!0,!0);var pos=getPosition(trigger,tip,conf);if(conf.tip&&tip.html(trigger.data("title")),e=$.Event(),e.type="onBeforeShow",fire.trigger(e,[pos]),e.isDefaultPrevented())return self;pos=getPosition(trigger,tip,conf),tip.css({position:"absolute",top:pos.top,left:pos.left}),shown=!0,effect[0].call(self,function(){e.type="onShow",shown="full",fire.trigger(e)});var event=conf.events.tooltip.split(/,\s*/);return tip.data("__set")||(tip.off(event[0]).on(event[0],function(){clearTimeout(timer),clearTimeout(pretimer)}),event[1]&&!trigger.is("input:not(:checkbox, :radio), textarea")&&tip.off(event[1]).on(event[1],function(e){e.relatedTarget!=trigger[0]&&trigger.trigger(evt[1].split(" ")[0])}),conf.tip||tip.data("__set",!0)),self},hide:function(e){return tip&&self.isShown()?(e=$.Event(),e.type="onBeforeHide",fire.trigger(e),e.isDefaultPrevented()?void 0:(shown=!1,effects[conf.effect][1].call(self,function(){e.type="onHide",fire.trigger(e)}),self)):self},isShown:function(fully){return fully?"full"==shown:shown},getConf:function(){return conf},getTip:function(){return tip},getTrigger:function(){return trigger}}),$.each("onHide,onBeforeShow,onShow,onBeforeHide".split(","),function(i,name){$.isFunction(conf[name])&&$(self).on(name,conf[name]),self[name]=function(fn){return fn&&$(self).on(name,fn),self}})}$.tools=$.tools||{version:"@VERSION"},$.tools.tooltip={conf:{effect:"toggle",fadeOutSpeed:"fast",predelay:0,delay:30,opacity:1,tip:0,fadeIE:!1,position:["top","center"],offset:[0,0],relative:!1,cancelDefault:!0,events:{def:"mouseenter,mouseleave",input:"focus,blur",widget:"focus mouseenter,blur mouseleave",tooltip:"mouseenter,mouseleave"},layout:"<div/>",tipClass:"tooltip"},addEffect:function(name,loadFn,hideFn){effects[name]=[loadFn,hideFn]}};var effects={toggle:[function(done){var conf=this.getConf(),tip=this.getTip(),o=conf.opacity;o<1&&tip.css({opacity:o}),tip.show(),done.call()},function(done){this.getTip().hide(),done.call()}],fade:[function(done){var conf=this.getConf();!/msie/.test(navigator.userAgent.toLowerCase())||conf.fadeIE?this.getTip().fadeTo(conf.fadeInSpeed,conf.opacity,done):(this.getTip().show(),done())},function(done){var conf=this.getConf();!/msie/.test(navigator.userAgent.toLowerCase())||conf.fadeIE?this.getTip().fadeOut(conf.fadeOutSpeed,done):(this.getTip().hide(),done())}]};$.fn.tooltip=function(conf){var api=this.data("tooltip");return api?api:(conf=$.extend(!0,{},$.tools.tooltip.conf,conf),"string"==typeof conf.position&&(conf.position=conf.position.split(/,?\s/)),this.each(function(){api=new Tooltip($(this),conf),$(this).data("tooltip",api)}),conf.api?api:this)}}(jQuery);

/* - nodeutilities.js - */
// https://www.gov.br/portal_javascripts/nodeutilities.js?original=1
function wrapNode(node,wrappertype,wrapperclass){jQuery(node).wrap('<'+wrappertype+'>').parent().addClass(wrapperclass)}
function nodeContained(innernode,outernode){return jQuery(innernode).parents().filter(function(){return this===outernode}).length>0}
function findContainer(node,func){var p=jQuery(node).parents().filter(func);return p.length?p.get(0):false}
function hasClassName(node,class_name){return jQuery(node).hasClass(class_name)}
function addClassName(node,class_name){jQuery(node).addClass(class_name)}
function removeClassName(node,class_name){jQuery(node).removeClass(class_name)}
function replaceClassName(node,old_class,new_class,ignore_missing){if(ignore_missing||jQuery(node).hasClass(old_class)){jQuery(node).removeClass(old_class).addClass(new_class)}}
function walkTextNodes(node,func,data){jQuery(node).find('*').andSelf().contents().each(function(){if(this.nodeType===3){func(this,data)}})}
function getInnerTextCompatible(node){return jQuery(node).text()}
function getInnerTextFast(node){return jQuery(node).text()}
function sortNodes(nodes,fetch_func,cmp_func){var SortNodeWrapper,items;SortNodeWrapper=function(node){this.value=fetch_func(node);this.cloned_node=node.cloneNode(true)};SortNodeWrapper.prototype.toString=function(){return this.value.toString?this.value.toString():this.value};items=jQuery(nodes).map(function(){return new SortNodeWrapper(this)});if(cmp_func){items.sort(cmp_func)} else{items.sort()}
jQuery.each(items, function(i){jQuery(nodes[i]).replace(this.cloned_node)})}
function copyChildNodes(srcNode,dstNode){jQuery(srcNode).children().clone().appendTo(jQuery(dstNode))}


/* - cookie_functions.js - */
// https://www.gov.br/portal_javascripts/cookie_functions.js?original=1
function createCookie(name,value,days){var date,expires;if(days){date=new Date();date.setTime(date.getTime()+(days*24*60*60*1000));expires="; expires="+date.toGMTString()} else{expires=""}
document.cookie=name+"="+escape(value)+expires+"; path=/;"}
function readCookie(name){var nameEQ=name+"=",ca=document.cookie.split(';'),i,c;for(i=0;i<ca.length;i=i+1){c=ca[i];while(c.charAt(0)===' '){c=c.substring(1,c.length)}
if(c.indexOf(nameEQ)===0){return unescape(c.substring(nameEQ.length,c.length))}}
return null}


/* - modernizr.js - */
// https://www.gov.br/portal_javascripts/modernizr.js?original=1
;window.Modernizr=function(a,b,c){function H(){e.input=function(a){for(var b=0,c=a.length;b<c;b++)t[a[b]]=a[b]in l;return t}("autocomplete autofocus list placeholder max min multiple pattern required step".split(" ")),e.inputtypes=function(a){for(var d=0,e,f,h,i=a.length;d<i;d++)l.setAttribute("type",f=a[d]),e=l.type!=="text",e&&(l.value=m,l.style.cssText="position:absolute;visibility:hidden;",/^range$/.test(f)&&l.style.WebkitAppearance!==c?(g.appendChild(l),h=b.defaultView,e=h.getComputedStyle&&h.getComputedStyle(l,null).WebkitAppearance!=="textfield"&&l.offsetHeight!==0,g.removeChild(l)):/^(search|tel)$/.test(f)||(/^(url|email)$/.test(f)?e=l.checkValidity&&l.checkValidity()===!1:/^color$/.test(f)?(g.appendChild(l),g.offsetWidth,e=l.value!=m,g.removeChild(l)):e=l.value!=m)),s[a[d]]=!!e;return s}("search tel url email datetime date month week time datetime-local number range color".split(" "))}function F(a,b){var c=a.charAt(0).toUpperCase()+a.substr(1),d=(a+" "+p.join(c+" ")+c).split(" ");return E(d,b)}function E(a,b){for(var d in a)if(k[a[d]]!==c)return b=="pfx"?a[d]:!0;return!1}function D(a,b){return!!~(""+a).indexOf(b)}function C(a,b){return typeof a===b}function B(a,b){return A(o.join(a+";")+(b||""))}function A(a){k.cssText=a}var d="2.0.4",e={},f=!0,g=b.documentElement,h=b.head||b.getElementsByTagName("head")[0],i="modernizr",j=b.createElement(i),k=j.style,l=b.createElement("input"),m=":)",n=Object.prototype.toString,o=" -webkit- -moz- -o- -ms- -khtml- ".split(" "),p="Webkit Moz O ms Khtml".split(" "),q={svg:"http://www.w3.org/2000/svg"},r={},s={},t={},u=[],v=function(a,c,d,e){var f,h,j,k=b.createElement("div");if(parseInt(d,10))while(d--)j=b.createElement("div"),j.id=e?e[d]:i+(d+1),k.appendChild(j);f=["&shy;","<style>",a,"</style>"].join(""),k.id=i,k.innerHTML+=f,g.appendChild(k),h=c(k,a),k.parentNode.removeChild(k);return!!h},w=function(){function d(d,e){e=e||b.createElement(a[d]||"div"),d="on"+d;var f=d in e;f||(e.setAttribute||(e=b.createElement("div")),e.setAttribute&&e.removeAttribute&&(e.setAttribute(d,""),f=C(e[d],"function"),C(e[d],c)||(e[d]=c),e.removeAttribute(d))),e=null;return f}var a={select:"input",change:"input",submit:"form",reset:"form",error:"img",load:"img",abort:"img"};return d}(),x,y={}.hasOwnProperty,z;!C(y,c)&&!C(y.call,c)?z=function(a,b){return y.call(a,b)}:z=function(a,b){return b in a&&C(a.constructor.prototype[b],c)};var G=function(c,d){var f=c.join(""),g=d.length;v(f,function(c,d){var f=b.styleSheets[b.styleSheets.length-1],h=f.cssRules&&f.cssRules[0]?f.cssRules[0].cssText:f.cssText||"",i=c.childNodes,j={};while(g--)j[i[g].id]=i[g];e.touch="ontouchstart"in a||j.touch.offsetTop===9,e.csstransforms3d=j.csstransforms3d.offsetLeft===9,e.generatedcontent=j.generatedcontent.offsetHeight>=1,e.fontface=/src/i.test(h)&&h.indexOf(d.split(" ")[0])===0},g,d)}(['@font-face {font-family:"font";src:url("https://")}',["@media (",o.join("touch-enabled),("),i,")","{#touch{top:9px;position:absolute}}"].join(""),["@media (",o.join("transform-3d),("),i,")","{#csstransforms3d{left:9px;position:absolute}}"].join(""),['#generatedcontent:after{content:"',m,'"}'].join("")],["fontface","touch","csstransforms3d","generatedcontent"]);r.flexbox=function(){function c(a,b,c,d){a.style.cssText=o.join(b+":"+c+";")+(d||"")}function a(a,b,c,d){b+=":",a.style.cssText=(b+o.join(c+";"+b)).slice(0,-b.length)+(d||"")}var d=b.createElement("div"),e=b.createElement("div");a(d,"display","box","width:42px;padding:0;"),c(e,"box-flex","1","width:10px;"),d.appendChild(e),g.appendChild(d);var f=e.offsetWidth===42;d.removeChild(e),g.removeChild(d);return f},r.canvas=function(){var a=b.createElement("canvas");return!!a.getContext&&!!a.getContext("2d")},r.canvastext=function(){return!!e.canvas&&!!C(b.createElement("canvas").getContext("2d").fillText,"function")},r.webgl=function(){return!!a.WebGLRenderingContext},r.touch=function(){return e.touch},r.geolocation=function(){return!!navigator.geolocation},r.postmessage=function(){return!!a.postMessage},r.websqldatabase=function(){var b=!!a.openDatabase;return b},r.indexedDB=function(){for(var b=-1,c=p.length;++b<c;)if(a[p[b].toLowerCase()+"IndexedDB"])return!0;return!!a.indexedDB},r.hashchange=function(){return w("hashchange",a)&&(b.documentMode===c||b.documentMode>7)},r.history=function(){return!!a.history&&!!history.pushState},r.draganddrop=function(){return w("dragstart")&&w("drop")},r.websockets=function(){for(var b=-1,c=p.length;++b<c;)if(a[p[b]+"WebSocket"])return!0;return"WebSocket"in a},r.rgba=function(){A("background-color:rgba(150,255,150,.5)");return D(k.backgroundColor,"rgba")},r.hsla=function(){A("background-color:hsla(120,40%,100%,.5)");return D(k.backgroundColor,"rgba")||D(k.backgroundColor,"hsla")},r.multiplebgs=function(){A("background:url(https://),url(https://),red url(https://)");return/(url\s*\(.*?){3}/.test(k.background)},r.backgroundsize=function(){return F("backgroundSize")},r.borderimage=function(){return F("borderImage")},r.borderradius=function(){return F("borderRadius")},r.boxshadow=function(){return F("boxShadow")},r.textshadow=function(){return b.createElement("div").style.textShadow===""},r.opacity=function(){B("opacity:.55");return/^0.55$/.test(k.opacity)},r.cssanimations=function(){return F("animationName")},r.csscolumns=function(){return F("columnCount")},r.cssgradients=function(){var a="background-image:",b="gradient(linear,left top,right bottom,from(#9f9),to(white));",c="linear-gradient(left top,#9f9, white);";A((a+o.join(b+a)+o.join(c+a)).slice(0,-a.length));return D(k.backgroundImage,"gradient")},r.cssreflections=function(){return F("boxReflect")},r.csstransforms=function(){return!!E(["transformProperty","WebkitTransform","MozTransform","OTransform","msTransform"])},r.csstransforms3d=function(){var a=!!E(["perspectiveProperty","WebkitPerspective","MozPerspective","OPerspective","msPerspective"]);a&&"webkitPerspective"in g.style&&(a=e.csstransforms3d);return a},r.csstransitions=function(){return F("transitionProperty")},r.fontface=function(){return e.fontface},r.generatedcontent=function(){return e.generatedcontent},r.video=function(){var a=b.createElement("video"),c=!1;try{if(c=!!a.canPlayType){c=new Boolean(c),c.ogg=a.canPlayType('video/ogg; codecs="theora"');var d='video/mp4; codecs="avc1.42E01E';c.h264=a.canPlayType(d+'"')||a.canPlayType(d+', mp4a.40.2"'),c.webm=a.canPlayType('video/webm; codecs="vp8, vorbis"')}}catch(e){}return c},r.audio=function(){var a=b.createElement("audio"),c=!1;try{if(c=!!a.canPlayType)c=new Boolean(c),c.ogg=a.canPlayType('audio/ogg; codecs="vorbis"'),c.mp3=a.canPlayType("audio/mpeg;"),c.wav=a.canPlayType('audio/wav; codecs="1"'),c.m4a=a.canPlayType("audio/x-m4a;")||a.canPlayType("audio/aac;")}catch(d){}return c},r.localstorage=function(){try{return!!localStorage.getItem}catch(a){return!1}},r.sessionstorage=function(){try{return!!sessionStorage.getItem}catch(a){return!1}},r.webworkers=function(){return!!a.Worker},r.applicationcache=function(){return!!a.applicationCache},r.svg=function(){return!!b.createElementNS&&!!b.createElementNS(q.svg,"svg").createSVGRect},r.inlinesvg=function(){var a=b.createElement("div");a.innerHTML="<svg/>";return(a.firstChild&&a.firstChild.namespaceURI)==q.svg},r.smil=function(){return!!b.createElementNS&&/SVG/.test(n.call(b.createElementNS(q.svg,"animate")))},r.svgclippaths=function(){return!!b.createElementNS&&/SVG/.test(n.call(b.createElementNS(q.svg,"clipPath")))};for(var I in r)z(r,I)&&(x=I.toLowerCase(),e[x]=r[I](),u.push((e[x]?"":"no-")+x));e.input||H(),e.addTest=function(a,b){if(typeof a=="object")for(var d in a)z(a,d)&&e.addTest(d,a[d]);else{a=a.toLowerCase();if(e[a]!==c)return;b=typeof b=="boolean"?b:!!b(),g.className+=" "+(b?"":"no-")+a,e[a]=b}return e},A(""),j=l=null,a.attachEvent&&function(){var a=b.createElement("div");a.innerHTML="<elem></elem>";return a.childNodes.length!==1}()&&function(a,b){function s(a){var b=-1;while(++b<g)a.createElement(f[b])}a.iepp=a.iepp||{};var d=a.iepp,e=d.html5elements||"abbr|article|aside|audio|canvas|datalist|details|figcaption|figure|footer|header|hgroup|mark|meter|nav|output|progress|section|summary|time|video",f=e.split("|"),g=f.length,h=new RegExp("(^|\\s)("+e+")","gi"),i=new RegExp("<(/*)("+e+")","gi"),j=/^\s*[\{\}]\s*$/,k=new RegExp("(^|[^\\n]*?\\s)("+e+")([^\\n]*)({[\\n\\w\\W]*?})","gi"),l=b.createDocumentFragment(),m=b.documentElement,n=m.firstChild,o=b.createElement("body"),p=b.createElement("style"),q=/print|all/,r;d.getCSS=function(a,b){if(a+""===c)return"";var e=-1,f=a.length,g,h=[];while(++e<f){g=a[e];if(g.disabled)continue;b=g.media||b,q.test(b)&&h.push(d.getCSS(g.imports,b),g.cssText),b="all"}return h.join("")},d.parseCSS=function(a){var b=[],c;while((c=k.exec(a))!=null)b.push(((j.exec(c[1])?"\n":c[1])+c[2]+c[3]).replace(h,"$1.iepp_$2")+c[4]);return b.join("\n")},d.writeHTML=function(){var a=-1;r=r||b.body;while(++a<g){var c=b.getElementsByTagName(f[a]),d=c.length,e=-1;while(++e<d)c[e].className.indexOf("iepp_")<0&&(c[e].className+=" iepp_"+f[a])}l.appendChild(r),m.appendChild(o),o.className=r.className,o.id=r.id,o.innerHTML=r.innerHTML.replace(i,"<$1font")},d._beforePrint=function(){p.styleSheet.cssText=d.parseCSS(d.getCSS(b.styleSheets,"all")),d.writeHTML()},d.restoreHTML=function(){o.innerHTML="",m.removeChild(o),m.appendChild(r)},d._afterPrint=function(){d.restoreHTML(),p.styleSheet.cssText=""},s(b),s(l);d.disablePP||(n.insertBefore(p,n.firstChild),p.media="print",p.className="iepp-printshim",a.attachEvent("onbeforeprint",d._beforePrint),a.attachEvent("onafterprint",d._afterPrint))}(a,b),e._version=d,e._prefixes=o,e._domPrefixes=p,e.hasEvent=w,e.testProp=function(a){return E([a])},e.testAllProps=F,e.testStyles=v,g.className=g.className.replace(/\bno-js\b/,"")+(f?" js "+u.join(" "):"");return e}(this,this.document),function(a,b,c){function k(a){return!a||a=="loaded"||a=="complete"}function j(){var a=1,b=-1;while(p.length- ++b)if(p[b].s&&!(a=p[b].r))break;a&&g()}function i(a){var c=b.createElement("script"),d;c.src=a.s,c.onreadystatechange=c.onload=function(){!d&&k(c.readyState)&&(d=1,j(),c.onload=c.onreadystatechange=null)},m(function(){d||(d=1,j())},H.errorTimeout),a.e?c.onload():n.parentNode.insertBefore(c,n)}function h(a){var c=b.createElement("link"),d;c.href=a.s,c.rel="stylesheet",c.type="text/css",!a.e&&(w||r)?function a(b){m(function(){if(!d)try{b.sheet.cssRules.length?(d=1,j()):a(b)}catch(c){c.code==1e3||c.message=="security"||c.message=="denied"?(d=1,m(function(){j()},0)):a(b)}},0)}(c):(c.onload=function(){d||(d=1,m(function(){j()},0))},a.e&&c.onload()),m(function(){d||(d=1,j())},H.errorTimeout),!a.e&&n.parentNode.insertBefore(c,n)}function g(){var a=p.shift();q=1,a?a.t?m(function(){a.t=="c"?h(a):i(a)},0):(a(),j()):q=0}function f(a,c,d,e,f,h){function i(){!o&&k(l.readyState)&&(r.r=o=1,!q&&j(),l.onload=l.onreadystatechange=null,m(function(){u.removeChild(l)},0))}var l=b.createElement(a),o=0,r={t:d,s:c,e:h};l.src=l.data=c,!s&&(l.style.display="none"),l.width=l.height="0",a!="object"&&(l.type=d),l.onload=l.onreadystatechange=i,a=="img"?l.onerror=i:a=="script"&&(l.onerror=function(){r.e=r.r=1,g()}),p.splice(e,0,r),u.insertBefore(l,s?null:n),m(function(){o||(u.removeChild(l),r.r=r.e=o=1,j())},H.errorTimeout)}function e(a,b,c){var d=b=="c"?z:y;q=0,b=b||"j",C(a)?f(d,a,b,this.i++,l,c):(p.splice(this.i++,0,a),p.length==1&&g());return this}function d(){var a=H;a.loader={load:e,i:0};return a}var l=b.documentElement,m=a.setTimeout,n=b.getElementsByTagName("script")[0],o={}.toString,p=[],q=0,r="MozAppearance"in l.style,s=r&&!!b.createRange().compareNode,t=r&&!s,u=s?l:n.parentNode,v=a.opera&&o.call(a.opera)=="[object Opera]",w="webkitAppearance"in l.style,x=w&&"async"in b.createElement("script"),y=r?"object":v||x?"img":"script",z=w?"img":y,A=Array.isArray||function(a){return o.call(a)=="[object Array]"},B=function(a){return typeof a=="object"},C=function(a){return typeof a=="string"},D=function(a){return o.call(a)=="[object Function]"},E=[],F={},G,H;H=function(a){function f(a){var b=a.split("!"),c=E.length,d=b.pop(),e=b.length,f={url:d,origUrl:d,prefixes:b},g,h;for(h=0;h<e;h++)g=F[b[h]],g&&(f=g(f));for(h=0;h<c;h++)f=E[h](f);return f}function e(a,b,e,g,h){var i=f(a),j=i.autoCallback;if(!i.bypass){b&&(b=D(b)?b:b[a]||b[g]||b[a.split("/").pop().split("?")[0]]);if(i.instead)return i.instead(a,b,e,g,h);e.load(i.url,i.forceCSS||!i.forceJS&&/css$/.test(i.url)?"c":c,i.noexec),(D(b)||D(j))&&e.load(function(){d(),b&&b(i.origUrl,h,g),j&&j(i.origUrl,h,g)})}}function b(a,b){function c(a){if(C(a))e(a,h,b,0,d);else if(B(a))for(i in a)a.hasOwnProperty(i)&&e(a[i],h,b,i,d)}var d=!!a.test,f=d?a.yep:a.nope,g=a.load||a.both,h=a.callback,i;c(f),c(g),a.complete&&b.load(a.complete)}var g,h,i=this.yepnope.loader;if(C(a))e(a,0,i,0);else if(A(a))for(g=0;g<a.length;g++)h=a[g],C(h)?e(h,0,i,0):A(h)?H(h):B(h)&&b(h,i);else B(a)&&b(a,i)},H.addPrefix=function(a,b){F[a]=b},H.addFilter=function(a){E.push(a)},H.errorTimeout=1e4,b.readyState==null&&b.addEventListener&&(b.readyState="loading",b.addEventListener("DOMContentLoaded",G=function(){b.removeEventListener("DOMContentLoaded",G,0),b.readyState="complete"},0)),a.yepnope=d()}(this,this.document),Modernizr.load=function(){yepnope.apply(window,[].slice.call(arguments,0))};

/* - livesearch.js - */
// https://www.gov.br/portal_javascripts/livesearch.js?original=1
var livesearch=(function(){var _2=400,_7=400,_0={},_1="LSHighlight";function _5(f,i){var l=null,r=null,c={},re=f.find('div.LSResult'),q=f.attr('action').replace(f.data('action')||'@@search',re.data('livesearch')||'livesearch_reply'),s=f.find('div.LSShadow'),p=f.find('input[name="path"]');function _12(){re.hide();l=null}function _6(){window.setTimeout('livesearch.hide("'+f.attr('id')+'")',_7)}function _11(d){re.show();s.html(d)}function _14(){var sort_on=function(){var parameters=location.search,sorton_position=parameters.indexOf('sort_on');if(sorton_position===-1){var s=$('#search-results');if(s.length>0){return s.attr('data-default-sort')}return ''}var sort_on=parameters.substring(sorton_position);sort_on=sort_on.split('&')[0];sort_on=sort_on.split('=')[1];return sort_on}();if(l===i.value){return}l=i.value;if(r&&r.readyState<4){r.abort()}if(i.value.length<2){_12();return}var qu={q:i.value};if(p.length&&p[0].checked){qu.path=p.val()}if(sort_on!==''){qu.sortOn=sort_on}qu=jQuery.param(qu);if(c[qu]){_11(c[qu]);return}r=jQuery.get(q,qu,function(d){_11(d);c[qu]=d},'text')}function _4(){window.setTimeout('livesearch.search("'+f.attr('id')+'")',_2)}return{hide:_12,hide_delayed:_6,search:_14,search_delayed:_4}}function _3(f){var t=null,re=f.find('div.LSResult'),s=f.find('div.LSShadow');function _16(){var c=s.find('li.LSHighlight').removeClass(_1),p=c.prev('li');if(!p.length){p=s.find('li:last')}p.addClass(_1);return false}function _9(){var c=s.find('li.LSHighlight').removeClass(_1),n=c.next('li');if(!n.length){n=s.find('li:first')}n.addClass(_1);return false}function _8(){s.find('li.LSHighlight').removeClass(_1);re.hide()}function _10(e){window.clearTimeout(t);switch(e.keyCode){case 38:return _16();case 40:return _9();case 27:return _8();case 37:break;case 39:break;default:t=window.setTimeout('livesearch.search("'+f.attr('id')+'")',_2)}}function _13(){var t=s.find('li.LSHighlight a').attr('href');if(!t){return}window.location=t;return false}return{handler:_10,submit:_13}}function _15(i){var i='livesearch'+i,f=jQuery(this).parents('form:first'),k=_3(f);_0[i]=_5(f,this);f.attr('id',i).submit(k.submit);jQuery(this).attr('autocomplete','off').keydown(k.handler).focus(_0[i].search_delayed).blur(_0[i].hide_delayed)}jQuery(function(){jQuery("#searchGadget,input.portlet-search-gadget").each(_15)});return{search:function(id){_0[id].search()},hide:function(id){_0[id].hide()}}}());

/* - select_all.js - */
// https://www.gov.br/portal_javascripts/select_all.js?original=1
function toggleSelect(selectbutton,id,initialState,formName){if (/MSIE [5-8]\./.test(navigator.userAgent)&&event.type==="change"&&/toggleSelect\(/.test(selectbutton.onchange.toString())){return}
var fid,state,base;fid=id||'ids:list';state=selectbutton.isSelected;if(state===undefined){state=Boolean(initialState)}
selectbutton.isSelected=!state;jQuery(selectbutton).attr('src',portal_url+'/select_'+(state?'all':'none')+'_icon.png');base=formName?jQuery(document.forms[formName]):jQuery(document);base.find('input[name="'+fid+'"]:checkbox').prop('checked',!state)}


/* - dragdropreorder.js - */
// https://www.gov.br/portal_javascripts/dragdropreorder.js?original=1
var ploneDnDReorder={};ploneDnDReorder.dragging=null;ploneDnDReorder.table=null;ploneDnDReorder.rows=null;ploneDnDReorder.locked=false;(function($){ploneDnDReorder.doDown=function(e){var dragging=ploneDnDReorder.dragging,body;if(ploneDnDReorder.locked){return}
if(dragging){if($(this).attr('id')!==dragging.attr('id')){ploneDnDReorder.locked=true;dragging.removeClass('dragging').addClass('error')}
return}
dragging=$(this).parents('.draggable:first');if(!dragging.length){return}
ploneDnDReorder.rows.mousemove(ploneDnDReorder.doDrag);body=$('body');body.mouseup(ploneDnDReorder.doUp);body.mouseleave(ploneDnDReorder.doCancel);ploneDnDReorder.dragging=dragging;dragging.data('ploneDnDReorder.startPosition',ploneDnDReorder.getPos(dragging));dragging.addClass("dragging");$(this).parents('tr').addClass('dragindicator');dragging.data('ploneDnDReorder.subset_ids',$.map(ploneDnDReorder.table.find('tr.draggable'),
function(elem){return $(elem).attr('id').substr('folder-contents-item-'.length)}));return false};ploneDnDReorder.getPos=function(node){var pos=node.parent().children('.draggable').index(node[0]);return pos===-1?null:pos};ploneDnDReorder.doDrag=function(e){var dragging=ploneDnDReorder.dragging,target=this;if(!dragging){return}
if(!target){return}
if($(target).attr('id')!==dragging.attr('id')){ploneDnDReorder.swapElements($(target),dragging)}
return false};ploneDnDReorder.swapElements=function(child1,child2){var parent=child1.parent(),items=parent.children('[id]'),t;if(Math.abs(ploneDnDReorder.getPos(child1)-ploneDnDReorder.getPos(child2))!==1){return}
items.removeClass('even').removeClass('odd');if(child1[0].swapNode){child1[0].swapNode(child2[0])} else{t=parent[0].insertBefore(document.createTextNode(''),child1[0]);child1.insertBefore(child2);child2.insertBefore(t);$(t).remove()}
parent.children('[id]:odd').addClass('even');parent.children('[id]:even').addClass('odd')};ploneDnDReorder.doUp=function(e){var dragging=ploneDnDReorder.dragging,body=$('body');if(!dragging){return}
ploneDnDReorder.updatePositionOnServer();dragging.removeData('ploneDnDReorder.startPosition');dragging.removeData('ploneDnDReorder.subset_ids');ploneDnDReorder.rows.unbind('mousemove',ploneDnDReorder.doDrag);body.unbind('mouseup',ploneDnDReorder.doUp);body.unbind('mouseleave',ploneDnDReorder.doCancel);$(this).parents('tr').removeClass('dragindicator');return false};ploneDnDReorder.doCancel=function(e){var dragging=ploneDnDReorder.dragging,body=$('body');if(!dragging){return}
dragging.removeClass("dragging");dragging.removeClass('dragindicator');if(ploneDnDReorder.getPos(dragging)-dragging.data('ploneDnDReorder.startPosition')){ploneDnDReorder.locked=true;dragging.addClass("error")}
ploneDnDReorder.rows.unbind('mousemove',ploneDnDReorder.doDrag);body.unbind('mouseup',ploneDnDReorder.doCancel);body.unbind('mouseleave',ploneDnDReorder.doCancel);ploneDnDReorder.dragging=null;return false};ploneDnDReorder.updatePositionOnServer=function(){var dragging=ploneDnDReorder.dragging,delta,args,encoded;if(!dragging){return}
delta=ploneDnDReorder.getPos(dragging)-dragging.data('ploneDnDReorder.startPosition');if(delta===0){ploneDnDReorder.doCancel.call();return}
args={item_id:dragging.attr('id').substr('folder-contents-item-'.length),subset_ids:dragging.data('ploneDnDReorder.subset_ids')};args['delta:int']=delta;encoded=$.param(args).replace(/%5B%5D=/g,'%3Alist=');$.ajax({type:'POST',url:'folder_moveitem',data:encoded,complete:ploneDnDReorder.complete});ploneDnDReorder.locked=true};ploneDnDReorder.complete=function(xhr,textStatus){var dragging=ploneDnDReorder.dragging;dragging.removeClass("dragging");dragging.removeClass('dragindicator');if(textStatus==="success"||textStatus==="notmodified"){ploneDnDReorder.locked=false} else{dragging.addClass("error")}
ploneDnDReorder.dragging=null}}(jQuery));
function initializeDnDReorder(table_selector){var table=table_selector;ploneDnDReorder.table=jQuery(table);if(!ploneDnDReorder.table.length)
return;ploneDnDReorder.rows=jQuery(table+" > tr,"+table+" > tbody > tr");jQuery(table+" > tr > td.draggable,"+table+" > tbody > tr > td.draggable").not('.notDraggable').mousedown(ploneDnDReorder.doDown).mouseup(ploneDnDReorder.doUp).addClass("draggingHook").css("cursor","ns-resize").html('&#x28ff;')}
$(document).ready(function(){initializeDnDReorder('#listing-table')});

/* - collapsiblesections.js - */
// https://www.gov.br/portal_javascripts/collapsiblesections.js?original=1
function activateCollapsibles(){(function($){$('dl.collapsible:not([class$=Collapsible])').find('dt.collapsibleHeader:first').click(function(){var c=$(this).parents('dl.collapsible:first'),t;if(!c){return true}t=c.hasClass('inline')?'Inline':'Block';c.toggleClass('collapsed'+t+'Collapsible').toggleClass('expanded'+t+'Collapsible')}).end().each(function(){var s=$(this).hasClass('collapsedOnLoad')?'collapsed':'expanded',t=$(this).hasClass('inline')?'Inline':'Block';$(this).removeClass('collapsedOnLoad').addClass(s+t+'Collapsible')})}(jQuery))}jQuery(function($){$(activateCollapsibles)});

/* - form_tabbing.js - */
// https://www.gov.br/portal_javascripts/form_tabbing.js?original=1
var ploneFormTabbing={jqtConfig:{current:'selected'},max_tabs:6};(function($){ploneFormTabbing._buildTabs=function(container,legends){var threshold=legends.length>ploneFormTabbing.max_tabs;var panel_ids,tab_ids=[],tabs='';for(var i=0;i<legends.length;i++){var className,tab,legend=legends[i],lid=legend.id;tab_ids[i]='#'+lid;switch(i){case(0):className='class="formTab firstFormTab"';break;case(legends.length-1):className='class="formTab lastFormTab"';break;default:className='class="formTab"';break}
if(threshold){tab='<option '+className+' id="'+lid+'" value="'+lid+'">';tab+=$(legend).text()+'</option>'} else{tab='<li '+className+'><a id="'+lid+'" href="';tab+=window.location.href+'#'+lid+'"><span>';tab+=$(legend).text()+'</span></a></li>'}
tabs+=tab;$(legend).css({'visibility':'hidden','font-size':'0','padding':'0','height':'0','width':'0','line-height':'0'})}
tab_ids=tab_ids.join(',');panel_ids=tab_ids.replace(/#fieldsetlegend-/g,"#fieldset-");if(threshold){tabs=$('<select class="formTabs">'+tabs+'</select>');tabs.change(function(){var selected=$(this).attr('value');$(this).parent().find('option#'+selected).click()})} else{tabs=$('<ul class="formTabs">'+tabs+'</ul>')}
return tabs};ploneFormTabbing.initializeDL=function(){var ftabs=$(ploneFormTabbing._buildTabs(this,$(this).children('dt')));var targets=$(this).children('dd');$(this).before(ftabs);targets.addClass('formPanel');ftabs.tabs(targets,ploneFormTabbing.jqtConfig)};ploneFormTabbing.initializeForm=function(){var jqForm=$(this);var fieldsets=jqForm.children('fieldset');if(!fieldsets.length){return}
var ftabs=ploneFormTabbing._buildTabs(this,fieldsets.children('legend'));$(this).prepend(ftabs);fieldsets.addClass("formPanel");$(this).find('input[name="fieldset"]').addClass('noUnloadProtection');$(this).find('.formPanel:has(div.field span.required)').each(function(){var id=this.id.replace(/^fieldset-/,"#fieldsetlegend-");$(id).addClass('required')});var initialIndex=0;var count=0;var found=false;$(this).find('.formPanel').each(function(){if(!found&&$(this).find('div.field.error').length!=0){initialIndex=count;found=true}
count+=1});var tabSelector='ul.formTabs';if($(ftabs).is('select.formTabs')){tabSelector='select.formTabs'}
var tabsConfig=$.extend({},ploneFormTabbing.jqtConfig,{'initialIndex':initialIndex});jqForm.children(tabSelector).tabs(jqForm.children('fieldset.formPanel'),tabsConfig);jqForm.submit(function(){var selected;if(ftabs.find('a.selected').length>=1){selected=ftabs.find('a.selected').attr('href').replace(/^#fieldsetlegend-/,"#fieldset-")}
else{selected=ftabs.attr('value').replace(/^fieldsetlegend-/,'#fieldset-')}
var fsInput=jqForm.find('input[name="fieldset"]');if(selected&&fsInput){fsInput.val(selected)}});$("#archetypes-schemata-links").addClass('hiddenStructure');$("div.formControls input[name='form.button.previous'],"+"div.formControls input[name='form.button.next']").remove()};$.fn.ploneTabInit=function(pbo){return this.each(function(){var item=$(this);item.find("form.enableFormTabbing,div.enableFormTabbing").each(ploneFormTabbing.initializeForm);item.find("dl.enableFormTabbing").each(ploneFormTabbing.initializeDL);var targetPane=window.location.hash||item.find('.enableFormTabbing input[name="fieldset"]').val();if(targetPane){item.find('.enableFormTabbing .formTabs [id="'+targetPane.replace('#','').replace('"','').replace(/^fieldset-/,"fieldsetlegend-")+'"]').click()}})};ploneFormTabbing.initialize=function(){$('body').ploneTabInit()}})(jQuery);jQuery(function(){ploneFormTabbing.initialize()});

/* - popupforms.js - */
// https://www.gov.br/portal_javascripts/popupforms.js?original=1
function msieversion(){var ua=window.navigator.userAgent;var msie=ua.indexOf("MSIE ");if(msie>0){return parseInt(ua.substring(msie+5,ua.indexOf(".",msie)))} else{return 0}}
var common_content_filter='#content>*:not(div.configlet),dl.portalMessage.error,dl.portalMessage.warning,dl.portalMessage.info';jQuery.extend(jQuery.tools.overlay.conf,{fixed:false,speed:'fast',mask:{color:'#fff',opacity:0.4,loadSpeed:0,closeSpeed:0}});(function($){$.plonepopups=$.plonepopups||{};$.extend($.plonepopups,{noformerrorshow: function noformerrorshow(el,noform){var o=$(el),emsg=o.find('dl.portalMessage.error');if(emsg.length){o.children().replaceWith(emsg);return false} else{return noform}},redirectbasehref: function redirectbasehref(el,responseText){var mo=responseText.match(/<base href="(\S+?)"/i);if(mo.length===2){return mo[1]}
return location}})})(jQuery);jQuery(function($){if(msieversion()>0&&msieversion()<7){return}
$('#portal-personaltools a[href$="/login"], #portal-personaltools a[href$="/login_form"], .discussion a[href$="/login"], .discussion a[href$="/login_form"]').prepOverlay({subtype:'ajax',filter:common_content_filter,formselector:'form#login_form',cssclass:'overlay-login',noform: function(){if(location.href.search(/pwreset_finish$/)>=0){return 'redirect'} else{return 'reload'}},redirect: function(){var href=location.href;if(href.search(/pwreset_finish$/)>=0){return href.slice(0,href.length-14)+'logged_in'} else{return href}}});$('#siteaction-contact a').prepOverlay({subtype:'ajax',filter:common_content_filter,cssclass:'overlay-contact',formselector:'form[name="feedback_form"]',noform: function(el){return $.plonepopups.noformerrorshow(el,'close')}});$('#contextSetDefaultPage, #folderChangeDefaultPage').prepOverlay({subtype:'ajax',filter:common_content_filter,cssclass:'overlay-default_view',formselector:'form[name="default_page_form"]',noform: function(el){return $.plonepopups.noformerrorshow(el,'reload')},closeselector:'[name="form.button.Cancel"]',width:'40%'});$('dl#plone-contentmenu-actions a#plone-contentmenu-actions-rename').prepOverlay({subtype:'ajax',filter:common_content_filter,cssclass:'overlay-rename',closeselector:'[name="form.button.Cancel"]',width:'40%'});$('dl#plone-contentmenu-factories a#plone-contentmenu-add-to-default-page').prepOverlay({subtype:'ajax',filter:common_content_filter,cssclass:'overlay-folder-factories',closeselector:'[name="form.button.Cancel"]',width:'40%'});$('#portal-personaltools a[href$="/@@register"]').prepOverlay({subtype:'ajax',filter:common_content_filter,cssclass:'overlay-register',formselector:'form.kssattr-formname-register'});$('form[name="users_add"], form[name="groups_add"]').prepOverlay({subtype:'ajax',filter:common_content_filter,cssclass:'overlay-users',formselector:'form.kssattr-formname-new-user, form[name="groups"]',noform: function(el){return $.plonepopups.noformerrorshow(el,'redirect')},redirect: function(){return location.href}});$('form[name="users_add"], form[name="groups_add"]').width($('input.add').outerWidth());$('form[name="users_add"] input.add, form[name="groups_add"] input.add').css('cursor','pointer');$('#content-history a').prepOverlay({subtype:'ajax',filter:'h2, #content-history',cssclass:'overlay-history',urlmatch:'@@historyview',urlreplace:'@@contenthistorypopup'})});

/* - jquery.highlightsearchterms.js - */
// https://www.gov.br/portal_javascripts/jquery.highlightsearchterms.js?original=1
(function($){var Highlighter,makeSearchKey,makeAddress,defaults;Highlighter=function(options){$.extend(this,options);this.terms=this.cleanTerms(this.terms.length?this.terms:this.getSearchTerms())};Highlighter.prototype={highlight: function(startnode){if(!this.terms.length||!startnode.length){return}
var self=this;$.each(this.terms, function(i,term){startnode.find('*:not(textarea)').andSelf().contents().each(function(){if(this.nodeType===3){self.highlightTermInNode(this,term)}})})},highlightTermInNode: function(node,word){var c=node.nodeValue,self=this,highlight,ci,index,next;if($(node).parent().hasClass(self.highlightClass)){return}
highlight=function(content){return $('<span class="'+self.highlightClass+'">&nbsp;</span>').text(content)};ci=self.caseInsensitive;while(c&&(index=(ci?c.toLowerCase():c).indexOf(word))>-1){$(node).before(document.createTextNode(c.substr(0,index))).before(highlight(c.substr(index,word.length))).before(document.createTextNode(c.substr(index+word.length)));next=node.previousSibling;$(node).remove();node=next;c=node.nodeValue}},queryStringValue: function(uri,regexp){var match,pair;if(uri.indexOf('?')<0){return ''}
uri=uri.substr(uri.indexOf('?')+1);while(uri.indexOf('=')>=0){uri=uri.replace(/^\&*/,'');pair=uri.split('&',1)[0];uri=uri.substr(pair.length);match=pair.match(regexp);if(match){return decodeURIComponent(match[match.length-1].replace(/\+/g,' '))}}
return ''},termsFromReferrer: function(){var ref,i,se;ref=$.fn.highlightSearchTerms._test_referrer!==null?$.fn.highlightSearchTerms._test_referrer:document.referrer;if(!ref){return ''}
for(i=0;i<this.referrers.length;i+=1){se=this.referrers[i];if(ref.match(se.address)){return this.queryStringValue(ref,se.key)}}
return ''},cleanTerms: function(terms){var self=this;return $.unique($.map(terms, function(term){term=$.trim(self.caseInsensitive?term.toLowerCase():term);return(!term||self.filterTerms.test(term))?null:term}))},getSearchTerms: function(){var terms=[],uri=$.fn.highlightSearchTerms._test_location!==null?$.fn.highlightSearchTerms._test_location:location.href;if(this.useReferrer){$.merge(terms,this.termsFromReferrer().split(/\s+/))}
if(this.useLocation){$.merge(terms,this.queryStringValue(uri,this.searchKey).split(/\s+/))}
return terms}};makeSearchKey=function(key){return(typeof key==='string')?new RegExp('^'+key+'=(.*)$','i'):key};makeAddress=function(addr){return(typeof addr==='string')?new RegExp('^https?://(www\\.)?'+addr,'i'):addr};$.fn.highlightSearchTerms=function(options){options=$.extend({},defaults,options);options=$.extend(options,{searchKey:makeSearchKey(options.searchKey),referrers:$.map(options.referrers, function(se){return{address:makeAddress(se.address),key:makeSearchKey(se.key)}})});if(options.includeOwnDomain){var hostname=$.fn.highlightSearchTerms._test_location!==null?$.fn.highlightSearchTerms._test_location:location.hostname;options.referrers.push({address:makeAddress(hostname.replace(/\./g,'\\.')),key:options.searchKey})}
new Highlighter(options).highlight(this);return this};$.fn.highlightSearchTerms.referrers=[{address:'google\\.',key:'q'},{address:'search\\.yahoo\\.',key:'p'},{address:'search\\.msn\\.',key:'q'},{address:'search\\.live\\.',key:'query'},{address:'search\\.aol\\.',key:'userQuery'},{address:'ask\\.com',key:'q'},{address:'altavista\\.',key:'q'},{address:'feedster\\.',key:'q'}];defaults={terms:[],useLocation:true,searchKey:'(searchterm|SearchableText)',useReferrer:true,referrers:$.fn.highlightSearchTerms.referrers,includeOwnDomain:true,caseInsensitive:true,filterTerms:/\b(not|and|or)\b/i,highlightClass:'highlightedSearchTerm'};$.fn.highlightSearchTerms._test_location=null;$.fn.highlightSearchTerms._test_referrer=null;$.fn.highlightSearchTerms._highlighter=Highlighter}(jQuery));jQuery(function($){$('#region-content,#content').highlightSearchTerms({includeOwnDomain:$('dl.searchResults').length===0})});

/* - first_input_focus.js - */
// https://www.gov.br/portal_javascripts/first_input_focus.js?original=1
jQuery(function($){if($("form div.error :input:first").focus().length){return}
$("form.enableAutoFocus :input:not(.formTabs):visible:first").focus()});

/* - accessibility.js - */
// https://www.gov.br/portal_javascripts/accessibility.js?original=1
function setBaseFontSize(f,r){var b=jQuery('body');if(r){b.removeClass('smallText').removeClass('largeText');createCookie("fontsize",f,365)}b.addClass(f)}jQuery(function($){var f=readCookie("fontsize");if(f){setBaseFontSize(f,0)}});

/* - styleswitcher.js - */
// https://www.gov.br/portal_javascripts/styleswitcher.js?original=1
function setActiveStyleSheet(title,reset){jQuery('link[rel*=style][title]').attr('disabled',true).find('[title='+title+']').attr('disabled',false);if(reset){createCookie("wstyle",title,365)}}
jQuery(function(){var style=readCookie("wstyle");if(style){setActiveStyleSheet(style,0)}});

/* - toc.js - */
// https://www.gov.br/portal_javascripts/toc.js?original=1
jQuery(function($){var dest,content,location,stack,oltoc,numdigits,wlh,target,targetOffset;dest=$('dl.toc dd.portletItem');content=$('#region-content,#content');if(!content||!dest.length){return}
dest.empty();location=window.location.href;if(window.location.hash){location=location.substring(0,location.lastIndexOf(window.location.hash))}
stack=[];$(content).find('*').not('.comment > h3').filter(function(){return (/^h[1234]$/).test(this.tagName.toLowerCase())}).not('.documentFirstHeading').each(function(i){var level,ol,li;level=this.nodeName.charAt(1);while(stack.length<level){ol=$('<ol>');if(stack.length){li=$(stack[stack.length-1]).children('li:last');if(!li.length){li=$('<li>').appendTo($(stack[stack.length-1]))}
li.append(ol)}
stack.push(ol)}
while(stack.length>level){stack.pop()}
$(this).before($('<a name="section-'+i+'" />'));$('<li>').append($('<a />').attr('href',location+'#section-'+i).text($(this).text())).appendTo($(stack[stack.length-1]))});if(stack.length){var oltoc=$(stack[0]);var i=1;while(oltoc.children('li').length==1){oltoc=$(stack[i]);i+=1}
if(i<=stack.length){$('dl.toc').show()}
numdigits=oltoc.children().length.toString().length;oltoc.addClass("TOC"+numdigits+"Digit");dest.append(oltoc);wlh=window.location.hash;if(wlh){target=$(wlh);target=target.length&&target||$('[name="'+wlh.slice(1)+'"]');targetOffset=target.offset();if(targetOffset){$('html,body').animate({scrollTop:targetOffset.top},0)}}}});

/* - collapsibleformfields.js - */
// https://www.gov.br/portal_javascripts/collapsibleformfields.js?original=1
(function($){$.fn.do_search_collapse=function(){
function check_used(element){var e=$(element);if(e.find('input[id$=_toggle]:checkbox').length>0){if(e.find('input[id$=_toggle]:checkbox:checked').length===0){return true}}
if(e.find(':text[value]').length>0){return true}
if(e.find('select .default_option').length>0){if(e.find('select .default_option:selected').length===0){return true}}
return false}
return this.each( function(){var indicator=$(this).find('.collapser:first'),collapse=$(this).find('.collapse:first');indicator.click(function(){var container=$(this).parent(),target=container.find('.collapse:first');target.slideToggle('normal');$(this).toggleClass('expanded');$(this).toggleClass('collapsed')});if(check_used(this)){indicator.addClass('expanded')} else{collapse.hide();indicator.addClass('collapsed')}})};jQuery(function($){$('.field.collapsible').do_search_collapse()})}(jQuery));

/* - ++resource++plone.app.discussion.javascripts/comments.js - */
// https://www.gov.br/portal_javascripts/++resource++plone.app.discussion.javascripts/comments.js?original=1
(function($){$.createReplyForm=function(comment_div){var comment_id=comment_div.attr("id");var reply_button=comment_div.find(".reply-to-comment-button");var reply_div=$("#commenting").clone(true);reply_div.find("#formfield-form-widgets-captcha").find("script").remove();reply_div.appendTo(comment_div).css("display","none");reply_div.removeAttr("id");$(reply_button).css("display","none");var reply_form=reply_div.find("form");reply_form.find("input[name='form.widgets.in_reply_to']").val(comment_id);var cancel_reply_button=reply_div.find(".cancelreplytocomment");cancel_reply_button.attr("id",comment_id);reply_form.find("input[name='form.buttons.cancel']").css("display","inline");reply_div.slideDown("slow");cancel_reply_button.css("display","inline")};$.clearForm=function(form_div){form_div.find(".error").removeClass("error");form_div.find(".fieldErrorBox").remove();form_div.find("input[type='text']").attr("value","");form_div.find("textarea").attr("value","")};$(window).load(function(){var post_comment_div=$("#commenting");var in_reply_to_field=post_comment_div.find("input[name='form.widgets.in_reply_to']");if(in_reply_to_field.val()!==""){var current_reply_id="#"+in_reply_to_field.val();var current_reply_to_div=$(".discussion").find(current_reply_id);$.createReplyForm(current_reply_to_div);$.clearForm(post_comment_div)}
$(".reply-to-comment-button").bind("click", function(e){var comment_div=$(this).parents().filter(".comment");$.createReplyForm(comment_div);$.clearForm(comment_div)});$("#commenting #form-buttons-cancel").bind("click", function(e){e.preventDefault();var reply_to_comment_button=$(this).
parents().
filter(".comment").
find(".reply-to-comment-button");$.reply_to_comment_form=$(this).parents().filter(".reply");$.reply_to_comment_form.slideUp("slow", function(){$(this).remove()});reply_to_comment_button.css("display","inline")});$("input[name='form.button.PublishComment']").on('click', function(){var trigger=this;var form=$(this).parents("form");var data=$(form).serialize();var form_url=$(form).attr("action");$.ajax({type:"GET",url:form_url,data:data,context:trigger,success: function(msg){form.find("input[name='form.button.PublishComment']").remove();form.parents(".state-pending").toggleClass('state-pending').toggleClass('state-published')},error: function(msg){return true}});return false});$("form[name='edit']").prepOverlay({cssclass:'overlay-edit-comment',width:'60%',subtype:'ajax',filter:'#content>*'})
$("input[name='form.button.DeleteComment']").on('click', function(){var trigger=this;var form=$(this).parents("form");var data=$(form).serialize();var form_url=$(form).attr("action");$.ajax({type:'POST',url:form_url,data:data,context:$(trigger).parents(".comment"),success: function(data){var comment=$(this);var clss=comment.attr('class');var treelevel=parseInt(clss[clss.indexOf('replyTreeLevel')+'replyTreeLevel'.length],10);var selector=".replyTreeLevel"+treelevel;for(var i=0;i<treelevel;i++){selector+=", .replyTreeLevel"+i}
comment.nextUntil(selector).each(function(){$(this).fadeOut('fast', function(){$(this).remove()})});$(this).fadeOut('fast', function(){$(this).remove()})},error: function(req,error){return true}});return false});$(".reply").find("input[name='form.buttons.reply']").css("display","none");$(".reply").find("input[name='form.buttons.cancel']").css("display","none");$(".reply-to-comment-button").css("display","inline")})}(jQuery));

/* - dropdown.js - */
// https://www.gov.br/portal_javascripts/dropdown.js?original=1
function hideAllMenus(){jQuery('dl.actionMenu').removeClass('activated').addClass('deactivated')}
function toggleMenuHandler(event){jQuery(this).parents('.actionMenu:first').toggleClass('deactivated').toggleClass('activated');return false}
function actionMenuDocumentMouseDown(event){if(jQuery(event.target).parents('.actionMenu:first').length){return true}
hideAllMenus()}
function actionMenuMouseOver(event){var menu_id=jQuery(this).parents('.actionMenu:first').attr('id'),switch_menu;if(!menu_id){return true}
switch_menu=jQuery('dl.actionMenu.activated').length>0;jQuery('dl.actionMenu').removeClass('activated').addClass('deactivated');if(switch_menu){jQuery('#'+menu_id).removeClass('deactivated').addClass('activated')}}
function initializeMenus(){jQuery(document).mousedown(actionMenuDocumentMouseDown);hideAllMenus();jQuery('dl.actionMenu dt.actionMenuHeader a').click(toggleMenuHandler).mouseover(actionMenuMouseOver);jQuery('dl.actionMenu > dd.actionMenuContent').click(hideAllMenus)}
jQuery(initializeMenus);
