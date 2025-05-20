const cookies_de_desempenho_id = 'cookies-de-desempenho';
const desempenho_domains = ['//www.google-analytics', '//www.googletagmanager'];
const cookies_de_terceiros_id = 'cookies-de-terceiros';
const terceiros_domains = [
    '//sipec.centralit',
    '//cse.google',
    '//www.doubleclick',
    '//www.google.com',
    '//connect.facebook',
    '//www.instagram.com',
    '//platform.twitter.com',
    '//www.facebook.com',
    '//open.spotify.com',
    '//siasg.centralit',
    '//chatbotatendimento.transportes.gov.br',
    '//servicos.dnit.gov.br/chatbot/chat',
    'webchat.botframework.com',
    '//www.linkedin.com'
];
const desempenho_domain_estatisticas = [
    'estatisticas.presidencia.gov.br',
    'matomo.iec.gov.br',
    'matomo.pf.gov.br',
    'GTM-NHZSKTD',
    'GTM-53LKPS5'
];
const terceiros_domain_estatisticas = [
    'connect.facebook.net',
    'static.ads-twitter.com',
    'snap.licdn.com',
    'Metasix'
];

function _scripts_handling(cookies_de_desempenho_selected, cookies_de_terceiros_selected) {

    var scripts = [...document.getElementsByTagName('script')];
    for (var i = 0; i < scripts.length; i++) {
        // desempenho
        if (!cookies_de_desempenho_selected) {
            var src = scripts[i].src;
            for (var index = 0; index < desempenho_domains.length; index++) {
                if (src.indexOf(desempenho_domains[index]) > -1) {
                    scripts[i].setAttribute('data-src', src);
                    scripts[i].removeAttribute('src');
                }
            }
            var html = scripts[i].innerHTML;
            for (var index = 0; index < desempenho_domain_estatisticas.length; index++) {
                if (html.indexOf(desempenho_domain_estatisticas[index]) > -1) {
                    scripts[i].setAttribute('type', 'text');
                }
            }
        } else {
            var data_src = false;
            if (scripts[i].dataset) {
                if ('src' in scripts[i].dataset) {
                    data_src = scripts[i].dataset.src;
                }
            }
            if (data_src) {
                for (var index = 0; index < desempenho_domains.length; index++) {
                    if (data_src.indexOf(desempenho_domains[index]) > -1) {
                        scripts[i].setAttribute('src', data_src);
                        scripts[i].removeAttribute('data-src');
                    }
                }
            }
            var html = scripts[i].innerHTML;
            for (var index = 0; index < desempenho_domain_estatisticas.length; index++) {
                if (html.indexOf(desempenho_domain_estatisticas[index]) > -1) {
                    var script_tag = document.createElement('script');
                    script_tag.type = 'text/javascript';
                    script_tag.text = html
                    scripts[i].parentNode.replaceChild(script_tag, scripts[i]);
                }
            }
        }

        // terceiros
        if (!cookies_de_terceiros_selected) {
            var src = scripts[i].src;
            for (var index = 0; index < terceiros_domains.length; index++) {
                if (src.indexOf(terceiros_domains[index]) > -1 && src.indexOf('recaptcha') == -1) {
                    scripts[i].setAttribute('data-src', src);
                    scripts[i].removeAttribute('src');
                }
            }
            var html = scripts[i].innerHTML;
            for (var index = 0; index < terceiros_domain_estatisticas.length; index++) {
                if (html.indexOf(terceiros_domain_estatisticas[index]) > -1) {
                    scripts[i].setAttribute('type', 'text');
                }
            }
        } else {
            var data_src = false;
            if (scripts[i].dataset) {
                if ('src' in scripts[i].dataset) {
                    data_src = scripts[i].dataset.src;
                }
            }
            if (data_src) {
                for (var index = 0; index < terceiros_domains.length; index++) {
                    if (data_src.indexOf(terceiros_domains[index]) > -1 && data_src.indexOf('recaptcha') == -1) {
                        scripts[i].setAttribute('src', data_src);
                        scripts[i].removeAttribute('data-src');
                    }
                }
            }
            var html = scripts[i].innerHTML;
            for (var index = 0; index < terceiros_domain_estatisticas.length; index++) {
                if (html.indexOf(terceiros_domain_estatisticas[index]) > -1) {
                    var script_tag = document.createElement('script');
                    script_tag.type = 'text/javascript';
                    script_tag.text = html
                    scripts[i].parentNode.replaceChild(script_tag, scripts[i]);
                }
            }
        }

    }

}


function _iframes_handling(cookies_de_desempenho_selected, cookies_de_terceiros_selected) {

    var iframes = document.getElementsByTagName('iframe');
    for (var i = 0; i < iframes.length; i++) {

        // desempenho
        if (!cookies_de_desempenho_selected) {
            var src = iframes[i].src;
            if (src) {
                for (var index = 0; index < desempenho_domains.length; index++) {
                    if (src.indexOf(desempenho_domains[index]) > -1) {
                        iframes[i].setAttribute('data-src', src);
                        iframes[i].removeAttribute('src');
                        iframes[i].classList.add('d-none');
                    }
                }
            }
        } else {
            var data_src = false;
            if (iframes[i].dataset) {
                if ('src' in iframes[i].dataset) {
                    data_src = iframes[i].dataset.src;
                }
            }
            if (data_src) {
                for (var index = 0; index < desempenho_domains.length; index++) {
                    if (data_src.indexOf(desempenho_domains[index]) > -1) {
                        iframes[i].setAttribute('src', data_src);
                        iframes[i].removeAttribute('data-src');
                        iframes[i].classList.remove('d-none');
                        iframes[i].contentWindow.location.reload();
                    }
                }
            }
        }

        // terceiros
        if (!cookies_de_terceiros_selected) {
            var src = iframes[i].src;
            if (src) {
                for (var index = 0; index < terceiros_domains.length; index++) {
                    if (src.indexOf(terceiros_domains[index]) > -1 && src.indexOf('recaptcha') == -1) {
                        iframes[i].setAttribute('data-src', src);
                        iframes[i].removeAttribute('src');
                        iframes[i].classList.add('d-none');
                        var parent_element = iframes[i].parentElement;
                        if (parent_element.classList.contains('video')) {
                            parent_element.classList.add('d-none');
                        }
                        var label_midia = document.querySelector('.label-midia');
                        if (label_midia) {
                            label_midia.classList.add('d-none');
                        }
                    }
                }
            }
        } else {
            var data_src = false;
            if (iframes[i].dataset) {
                if ('src' in iframes[i].dataset) {
                    data_src = iframes[i].dataset.src;
                }
            }
            if (data_src) {
                for (var index = 0; index < terceiros_domains.length; index++) {
                    if (data_src.indexOf(terceiros_domains[index]) > -1 == -1) {
                        iframes[i].setAttribute('src', data_src);
                        iframes[i].removeAttribute('data-src');
                        iframes[i].classList.remove('d-none');
                        var parent_element = iframes[i].parentElement;
                        if (parent_element.classList.contains('d-none')) {
                            parent_element.classList.remove('d-none');
                        }
                        var label_midia = document.querySelector('.label-midia.d-none');
                        if (label_midia) {
                            label_midia.classList.remove('d-none');
                        }
                        iframes[i].contentWindow.location.reload();
                    }
                }
            }
        }

    }
}


function _cookies_handling(cookies_de_desempenho_selected, cookies_de_terceiros_selected) {
    if (!cookies_de_desempenho_selected) {
        var cookies_list = _get_cookie_list(cookies_de_desempenho_id);
        for (var i = 0; i < cookies_list.length; i++) {
            cookie_id = cookies_list[i];
            _delete_cookie(cookie_id);
        }
    }
    if (!cookies_de_terceiros_selected) {
        var cookies_list = _get_cookie_list(cookies_de_terceiros_id);
        for (var i = 0; i < cookies_list.length; i++) {
            cookie_id = cookies_list[i];
            _delete_cookie(cookie_id)
        }
    }
}


function _get_lgpd_cookie_v2() {
    var value = ('; ' + document.cookie).split(`; lgpd-cookie-v2=`).pop().split(';')[0];
    if (value) {
        return JSON.parse(value);
    }
    return false;
}


function _get_cookie_group(name) {
    var lgpd = _get_lgpd_cookie_v2();
    if (lgpd) {
        var groups = lgpd['g'];
        for (var i = 0, l = groups.length; i < l; i++) {
            if (groups[i]['id'] == name) {
                const cookies =  JSON.parse(localStorage.getItem(groups[i]['id']));
                return { 'groupId': groups[i]['id'], 'groupSelected':  groups[i]['on'], 'cookieList':  cookies };
            }
        }
    }
    // padrao not selected
    return { 'groupId': name, 'groupSelected': false, 'cookieList': [] };
}


function _get_cookie_list(name) {
    var group = _get_cookie_group(name);
    return group['cookieList'];
}


function _is_cookie_group_selected(name) {
    var group = _get_cookie_group(name);
    return group['groupSelected'];
}


function _delete_cookie(name) {
    document.cookie = name + '=; expires=Thu, 01 Jan 2000 00:00:01 GMT; path=/';
    // para remover os cookie do analytics nos ambientes de homologacao, funciona apenas informando o domain
    document.cookie = name + '=; expires=Thu, 01 Jan 2000 00:00:01 GMT; path=/; domain=.serpro.gov.br';
    // para remover os cookie do analytics nos ambientes de producao, funciona apenas informando o domain
    document.cookie = name + '=; expires=Thu, 01 Jan 2000 00:00:01 GMT; path=/; domain=.www.gov.br';
}


function _fb_page(cookies_de_terceiros_selected) {

    var div = document.querySelectorAll('div.fb-page').forEach((div) => {

        if (!cookies_de_terceiros_selected) {
            var data_href = false;
            if (div.dataset) {
                if ('href' in div.dataset) {
                    data_href = div.dataset.href;
                }
            }
            if (data_href) {
                div.setAttribute('data-dthref', data_href);
                div.removeAttribute('data-href');
            }
        } else {
            var data_href = false;
            if (div.dataset) {
                if ('dthref' in div.dataset) {
                    data_href = div.dataset.dthref;
                }
            }
            if (data_href) {
                div.setAttribute('data-href', data_href);
                div.removeAttribute('data-dthref');
            }
        }

    })
}


jQuery(document).ready(function() {
    // cookies-de-desempenho
    var cookies_de_desempenho_selected = _is_cookie_group_selected(cookies_de_desempenho_id);
    // cookies-de-terceiros
    var cookies_de_terceiros_selected = _is_cookie_group_selected(cookies_de_terceiros_id);
    // cookies
    _cookies_handling(cookies_de_desempenho_selected, cookies_de_terceiros_selected);
    // <script
    _scripts_handling(cookies_de_desempenho_selected, cookies_de_terceiros_selected);
    // <iframe
    _iframes_handling(cookies_de_desempenho_selected, cookies_de_terceiros_selected);
    // <div class="fb-page"
    _fb_page(cookies_de_terceiros_selected);
});
