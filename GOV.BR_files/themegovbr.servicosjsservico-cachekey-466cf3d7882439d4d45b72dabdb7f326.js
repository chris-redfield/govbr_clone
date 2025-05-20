
/* - ++theme++govbr.servicos/js/servico.js - */
// https://www.gov.br/portal_javascripts/++theme++govbr.servicos/js/servico.js?original=1
function load_recaptcha(){var url=window.location.href.split('#')[0];$.ajax({url:url+"/@@recaptcha",type:"GET",dataType:"text",success: function(result){$("#recaptcha").html(result)},error: function(result){$("recaptcha").html("Erro ao carregar recaptcha.")}})}

