/* 本地 HTML 自动连接完整网站；通过 HTTP 访问时保持当前页面。 */
(function () {
    'use strict';
    if (window.location.protocol !== 'file:') return;

    const site = 'http://127.0.0.1:5000/';
    let connecting = false;
    let connected = false;
    let panel;
    let message = '正在连接本机网站…';

    function showStatus(text) {
        message = text;
        if (panel) panel.querySelector('[data-status]').textContent = text;
    }

    async function connect() {
        if (connecting || connected) return;
        connecting = true;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);
        try {
            const response = await fetch(site + 'api/ping', {
                signal: controller.signal, cache: 'no-store'
            });
            const data = await response.json();
            if (!response.ok || data.service !== 'opd-web' || data.status !== 'ok') {
                throw new Error('wrong-service');
            }
            connected = true;
            showStatus('连接成功，正在打开完整网站…');
            window.location.replace(site + window.location.search + window.location.hash);
        } catch (error) {
            showStatus(error.message === 'wrong-service'
                ? '5000 端口上的服务不是本项目。请先关闭占用端口的程序，再启动网站。'
                : '暂未连接到网站。启动服务后，本页会自动进入完整网站。');
        } finally {
            clearTimeout(timeout);
            connecting = false;
            if (!connected) setTimeout(connect, 3000);
        }
    }

    function showEntry() {
        if (connected) return;
        panel = document.createElement('div');
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        panel.setAttribute('aria-labelledby', 'websiteEntryTitle');
        panel.style.cssText = 'position:fixed;inset:0;z-index:20000;display:flex;align-items:center;justify-content:center;background:#edf3faf5;padding:24px;overflow:auto';
        panel.innerHTML = `
            <section style="max-width:560px;background:white;padding:32px;border-radius:18px;box-shadow:0 15px 60px #19345322;font:16px/1.8 'Microsoft YaHei',sans-serif;color:#24344b">
                <h1 id="websiteEntryTitle" style="font-size:24px;margin:0 0 16px">打开 OCTree 完整网站</h1>
                <p data-status aria-live="polite"></p>
                <p style="margin:16px 0">首次使用或电脑重启后，请在此 HTML 所在文件夹双击 <strong>一键启动.bat</strong>。它会启动服务并自动打开浏览器，无需输入命令。</p>
                <p style="margin:16px 0">服务运行后，每次打开此 HTML 都会自动进入网站，使用表格上传、模拟测试、Math 和 Code 功能。</p>
                <a href="${site}" style="display:inline-block;background:#0666cc;color:white;padding:10px 20px;border-radius:8px;text-decoration:none">进入完整网站</a>
                <button type="button" data-retry style="margin-left:12px;padding:10px 16px;cursor:pointer">重新连接</button>
                <p style="margin-top:16px;font-size:14px;color:#57657a">若浏览器拦截本地连接检测，可直接点击“进入完整网站”。Linux 用户可运行同目录的“一键启动.sh”。</p>
            </section>`;
        document.body.appendChild(panel);
        showStatus(message);
        panel.querySelector('[data-retry]').addEventListener('click', connect);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', showEntry, {once: true});
    } else {
        showEntry();
    }
    connect();
})();
