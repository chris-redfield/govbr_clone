
/* - ++resource++govbr.policy/folder_contents.js - */
// https://www.gov.br/portal_javascripts/++resource++govbr.policy/folder_contents.js?original=1
function FolderContentsFunction(){if($('body.template-folder_contents').length){var botao=$("input[name^='content_status_history']");botao.on("click",function(e){var qtd=$(":checkbox:checked").length;if(qtd>20){if(!confirm('ATENÇÃO: Alterar o estado de mais de 20 itens simultaneamente pode gerar lentidão ou indisponibilidade na plataforma. Tem certeza que deseja prosseguir?')){return false}}
return true})}
if($('body.template-content_status_history').length){$('body.template-content_status_history .formSingleCheckbox #include_children').click(function(){if($(this).is(':checked')){if(!confirm('ATENÇÃO: Caso as pastas selecionadas contenham muitos itens, alterar o estado de todos simultaneamente pode gerar lentidão ou indisponibilidade na plataforma. Tem certeza que deseja prosseguir?')){return false}}})}}
jQuery(document).ready(function(){FolderContentsFunction()});jQuery(document).ajaxComplete(function(){FolderContentsFunction()});
