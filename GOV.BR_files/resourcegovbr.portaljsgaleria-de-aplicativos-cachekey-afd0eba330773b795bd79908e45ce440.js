
/* Merged Plone Javascript file
 * This file is dynamically assembled from separate parts.
 * Some of these parts have 3rd party licenses or copyright information attached
 * Such information is valid for that section,
 * not for the entire composite file
 * originating files are separated by - filename.js -
 */

/* - ++resource++govbr.portal/js/galeria-de-aplicativos.js - */
// https://www.gov.br/portal_javascripts/++resource++govbr.portal/js/galeria-de-aplicativos.js?original=1
function govbr_galeria_de_aplicativos_formwidget_autocomplete_ready(event,data,formatted){(function($){var input_box=$(event.target);input_box.attr("value",data[1]);$("#galeria_de_aplicativos").submit()})(jQuery)}(function($){$(document).ready(function(){if($("form#galeria_de_aplicativos").length){$('input[name="pesquisar_em_aplicativos"]').focus(function(){if($(this).attr("placeholder")==$(this).attr("value")){$(this).attr("value","")}})}})})(jQuery);

/* - ++resource++govbr.portal/js/noticias_ebc_config.js - */
// https://www.gov.br/portal_javascripts/++resource++govbr.portal/js/noticias_ebc_config.js?original=1
$(document).ready(function(){desabilita_campo("#form-widgets-log_ultima_atualizacao_capa");desabilita_campo("#form-widgets-dados_capa_json");desabilita_campo("#form-widgets-log_ultima_atualizacao_template");desabilita_campo("#form-widgets-dados_template_json");desabilita_campo("#form-widgets-blocos_validos_template")});
function desabilita_campo(element){$(element).focusin(function(){$(this).blur()})};
