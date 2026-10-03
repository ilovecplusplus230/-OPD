/* 表格样例测试、上传与报告共用一个入口。 */
(function () {
    'use strict';

    let busy = false;
    let initialized = false;
    let report = null;
    let chart = null;
    const byId = id => document.getElementById(id);
    const metricNames = {accuracy: 'Accuracy', balanced_accuracy: 'Balanced Accuracy', auc: 'AUC',
        f1: 'Weighted F1', f1_macro: 'Macro F1', log_loss: 'Log Loss', mse: 'MSE', rmse: 'RMSE', mae: 'MAE', r2: 'R²'};
    const lowerIsBetter = key => ['mse', 'rmse', 'mae', 'log_loss'].includes(key);
    const applicableMetrics = data => Object.entries(metricNames).filter(([key]) =>
        numeric(data.baseline[key]) !== null || numeric(data.optimized[key]) !== null);

    function numeric(value) {
        if (value === null || value === undefined || value === '') return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function format(value, digits = 4) {
        const number = numeric(value);
        if (number === null) return '不适用';
        return number !== 0 && Math.abs(number) < 0.0001 ? number.toExponential(2) : number.toFixed(digits);
    }

    const displayJson = value => JSON.stringify(value, (key, item) =>
        typeof item === 'number' && !Number.isInteger(item) ? format(item) : item, 2);

    function explanationText(proposal) {
        const adaptation = proposal.adaptation || {};
        const reference = adaptation.source_candidate ? `\n反馈来源：第 ${adaptation.source_round} 轮 ${adaptation.source_candidate}（${adaptation.outcome}）\n${adaptation.observed_reason || ''}` : '';
        const constants = (proposal.constant_provenance || []).map(item =>
            `${item.symbol} ≈ ${format(item.value)} [${item.source}]：${item.calculation} ${item.purpose}`).join('\n');
        return `历史反馈调整：${adaptation.action || '未记录'}${reference}\n参数来源与计算：\n${constants}\n公式构造步骤：\n${(proposal.derivation || []).join('\n')}`;
    }

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, char => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[char]));
    }

    function relativeChange(before, after, lowerIsBetter = false) {
        before = numeric(before);
        after = numeric(after);
        if (before === null || after === null) return '不适用';
        const gain = lowerIsBetter ? before - after : after - before;
        if (gain === 0) return '无变化';
        if (before === 0) return `变化量 ${format(after - before)}（基准为 0）`;
        const percent = gain / Math.abs(before) * 100;
        return `${percent > 0 ? '改善' : '下降'} ${Math.abs(percent).toFixed(2)}%`;
    }

    function normalizeResult(payload) {
        const record = payload.record || payload;
        const generated = record.generated || {};
        const verification = record.verification || {};
        const model = payload.model || verification.model || {};
        const baseline = payload.baseline || verification.baseline || {};
        const optimized = payload.optimized || verification.optimized || {};
        const acceptedFeatures = payload.accepted_features || generated.accepted_features || [];
        let current = {...baseline};
        const iterations = (payload.iterations || generated.iterations || []).map((item, index) => {
            const before = item.metrics_before || {...current};
            const after = item.metrics_after || (item.accepted ? item.metrics : before) || before;
            current = {...after};
            return {
                round: item.round ?? index + 1,
                code: item.feature_code || '',
                accepted: Boolean(item.accepted),
                reason: item.reason || '',
                metrics: item.valid === false ? {} : item.metrics || {},
                explanation: item.explanation || {},
                candidates: item.candidates || [],
                before,
                after
            };
        });
        return {
            baseline, optimized, iterations, acceptedFeatures,
            tree: payload.decision_tree_explanation || (record.structured_representation || {}).decision_tree_rules || '',
            summary: [
                payload.message || `分析完成：共评估 ${iterations.length} 轮，保留 ${acceptedFeatures.length} 个新特征。`,
                model.backend ? `评估模型：XGBoost（${String(model.device || '').startsWith('cuda') ? 'GPU' : 'CPU'}）。` : ''
            ].filter(Boolean).join(' ')
        };
    }

    function metricCards(data) {
        return applicableMetrics(data).map(([key, label]) => {
            const change = relativeChange(data.baseline[key], data.optimized[key], lowerIsBetter(key));
            return `<div class="metric-card"><div class="metric-value">${format(data.baseline[key], 3)} → ${format(data.optimized[key], 3)}</div><div class="metric-label">${label}</div><div class="metric-change">${change}</div></div>`;
        }).join('') + `<div class="metric-card"><div class="metric-value">${data.acceptedFeatures.length}</div><div class="metric-label">保留的新特征</div></div>`;
    }

    function ruleCards(data) {
        if (!data.iterations.length) return '<p>没有迭代记录。</p>';
        return data.iterations.flatMap(round => round.candidates.length ? round.candidates.map((candidate, index) => ({
            round: `${round.round} · 候选 ${index + 1}`, code: candidate.feature_code || '',
            accepted: candidate.accepted, reason: candidate.reason || '', metrics: candidate.metrics || {},
            explanation: candidate.proposal || {}, usage: candidate.model_usage || {}
        })) : [round]).map(item => {
            const state = item.accepted ? 'accepted' : 'rejected';
            const metrics = Object.entries(metricNames).filter(([key]) => numeric(item.metrics[key]) !== null)
                .map(([key, label]) => `${label}=${format(item.metrics[key])}`).join('，');
            const explanation = item.explanation || {};
            const details = explanation.hypothesis ? `<p>来源：${escapeHtml(explanation.source || '')}；输入列：${escapeHtml((explanation.input_columns || []).join('、'))}</p><p>关系假说：${escapeHtml(explanation.hypothesis)}</p><p>提取方式：${escapeHtml(explanation.construction)}</p><pre style="white-space:pre-wrap">${escapeHtml(explanationText(explanation))}</pre><details><summary>训练集统计依据（显示近似值）</summary><pre>${escapeHtml(displayJson(explanation.evidence || {}))}</pre></details>` : '';
            const usage = Object.entries(item.usage || {}).map(([name, info]) => `${name}：${info.split_count || 0} 次分裂`).join('；');
            const formula = explanation.display_expression || item.code;
            return `<div class="rule-item rule-${state}"><div class="rule-header"><span class="rule-round">第 ${escapeHtml(item.round)} 轮</span><span class="rule-status status-${state}">${item.accepted ? '已接受' : '未接受'}</span></div>${details}<pre class="rule-code">${escapeHtml(formula)}</pre><div class="rule-metrics">模型使用：${escapeHtml(usage || '未记录')}</div><div class="rule-metrics">候选评估：${escapeHtml(metrics || '未进入模型评估')}</div><div class="rule-metrics">${escapeHtml(item.reason)}</div></div>`;
        }).join('');
    }

    function renderChart(data) {
        if (chart) { chart.destroy(); chart = null; }
        const canvas = byId('iterationChartCanvas');
        let fallback = byId('tabularChartFallback');
        if (!fallback) {
            fallback = document.createElement('div');
            fallback.id = 'tabularChartFallback';
            canvas.parentNode.appendChild(fallback);
        }
        const points = [data.baseline, ...data.iterations.map(item => item.after)];
        const labels = ['基线', ...data.iterations.map(item => `第 ${item.round} 轮`)];
        const primary = points.some(point => numeric(point.f1_macro) !== null) ? 'f1_macro'
            : points.some(point => numeric(point.f1) !== null) ? 'f1' : 'mse';
        canvas.hidden = true;
        fallback.hidden = false;
        fallback.innerHTML = `<table><thead><tr><th>轮次</th><th>${metricNames[primary]}</th></tr></thead><tbody>${points.map((point, index) => `<tr><td>${escapeHtml(labels[index])}</td><td>${format(point[primary])}</td></tr>`).join('')}</tbody></table>`;
        if (typeof window.Chart !== 'function') return;
        const datasets = [];
        if (primary !== 'mse') {
            datasets.push({label: metricNames[primary], data: points.map(point => numeric(point[primary])), borderColor: '#0066cc', yAxisID: 'y'});
        }
        if (points.some(point => numeric(point.mse) !== null)) {
            datasets.push({label: 'MSE', data: points.map(point => numeric(point.mse)), borderColor: '#dc2626', yAxisID: 'y1'});
        }
        try {
            canvas.hidden = false;
            chart = new window.Chart(canvas, {
                type: 'line',
                data: {labels, datasets},
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        y: {display: datasets.some(item => item.yAxisID === 'y'), min: 0, max: 1},
                        y1: {display: datasets.some(item => item.yAxisID === 'y1'), position: 'right', min: 0, grid: {drawOnChartArea: false}}
                    }
                }
            });
            fallback.hidden = true;
        } catch (error) {
            canvas.hidden = true;
            console.warn('图表不可用，保留指标表格。', error);
        }
    }

    function showReport(payload) {
        report = normalizeResult(payload);
        byId('analysisModal').style.display = 'flex';
        byId('loadingArea').style.display = 'none';
        byId('resultArea').style.display = 'block';
        byId('metricsRow').innerHTML = metricCards(report);
        byId('treeContent').textContent = report.tree || '没有决策树规则。';
        byId('rulesList').innerHTML = ruleCards(report);
        byId('summaryRow').textContent = report.summary;
        byId('metric-iter').textContent = String(report.iterations.length);
        byId('metric-rules').textContent = String(report.acceptedFeatures.length);
        byId('console').textContent = [
            report.summary,
            ...applicableMetrics(report).map(([key, label]) => `${label}：${format(report.baseline[key])} → ${format(report.optimized[key])}；${relativeChange(report.baseline[key], report.optimized[key], lowerIsBetter(key))}`),
            ...report.iterations.flatMap(item => item.candidates.length ? item.candidates.map(candidate => {
                const proposal = candidate.proposal || {};
                return `第 ${item.round} 轮候选：${proposal.name || ''}\n依据列：${(proposal.input_columns || []).join('、')}\n关系假说：${proposal.hypothesis || ''}\n提取方式：${proposal.construction || ''}\n${explanationText(proposal)}\n${proposal.display_expression || candidate.feature_code}\n${candidate.accepted ? '接受' : '未接受'}：${candidate.reason}`;
            }) : [`第 ${item.round} 轮：${item.accepted ? '接受' : '未接受'}\n${item.code}\n${item.reason}`])
        ].join('\n');
        renderChart(report);
    }

    function closeAnalysisModal() {
        byId('analysisModal').style.display = 'none';
        if (chart) { chart.destroy(); chart = null; }
    }

    function downloadReport() {
        if (!report) return;
        const content = `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OCTree 表格报告</title><body><h1>OCTree 表格分析报告</h1><p>${escapeHtml(report.summary)}</p>${metricCards(report)}<h2>迭代记录</h2>${ruleCards(report)}<h2>决策树规则</h2><pre>${escapeHtml(report.tree)}</pre></body></html>`;
        const url = URL.createObjectURL(new Blob([content], {type: 'text/html;charset=utf-8'}));
        const link = document.createElement('a');
        link.href = url;
        link.download = `OCTree_Report_${new Date().toISOString().slice(0, 10)}.html`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    async function runRequest(url, options, label) {
        if (busy) return;
        busy = true;
        report = null;
        const controls = [...document.querySelectorAll('#runSimulateBtn, #uploadDataBtn, #fileInput, .final-demo-btn')];
        const states = controls.map(control => control.disabled);
        controls.forEach(control => { control.disabled = true; });
        byId('analysisModal').style.display = 'flex';
        byId('loadingArea').style.display = 'flex';
        byId('resultArea').style.display = 'none';
        byId('progressBar').style.width = '100%';
        byId('progressPercent').textContent = '处理中';
        byId('loadingStep').textContent = label;
        byId('console').textContent = label;
        byId('metric-iter').textContent = '运行中';
        byId('metric-rules').textContent = '—';
        try {
            const response = await fetch(url, options);
            const payload = await response.json();
            if (!response.ok || payload.success === false) throw new Error(payload.error || `HTTP ${response.status}`);
            showReport(payload);
            return payload;
        } catch (error) {
            closeAnalysisModal();
            byId('metric-iter').textContent = '失败';
            byId('metric-rules').textContent = '—';
            byId('console').textContent = `分析失败：${error.message}`;
            throw error;
        } finally {
            busy = false;
            controls.forEach((control, index) => { control.disabled = states[index]; });
        }
    }

    function runSimulation() {
        return runRequest('/api/simulate', {method: 'POST'}, '正在用Jungle Chess 训练集样例和本地特征规则运行交叉验证...');
    }

    function runDemo() {
        return runRequest('/api/run/tabular', {method: 'POST'}, '正在运行表格特征扩充与模型评估...');
    }

    async function uploadAndAnalyze() {
        const input = byId('fileInput');
        const file = input.files && input.files[0];
        if (!file || busy) return;
        try {
            if (!/\.(csv|xlsx|xls)$/i.test(file.name)) throw new Error('请上传 CSV 或 Excel 文件。');
            if (file.size > 50 * 1024 * 1024) throw new Error('文件大小不能超过 50MB。');
            const form = new FormData();
            form.append('file', file);
            await runRequest('/api/upload', {method: 'POST', body: form}, `正在分析 ${file.name}...`);
        } catch (error) {
            byId('console').textContent = `分析失败：${error.message}`;
        } finally {
            input.value = '';
        }
    }

    function init() {
        if (initialized) return;
        initialized = true;
        for (const id of ['uploadDataBtn', 'medicalOption', 'generalOption']) {
            const control = byId(id);
            if (control) control.addEventListener('click', () => { if (!busy) byId('fileInput').click(); });
        }
        byId('fileInput').addEventListener('change', uploadAndAnalyze);
        byId('runSimulateBtn').addEventListener('click', () => { runSimulation().catch(() => {}); });
    }

    window.TabularUI = {
        normalizeResult, format, relativeChange, showReport, runDemo, runSimulation,
        isBusy: () => busy,
        refreshChart: () => { if (report) renderChart(report); }
    };
    window.downloadReport = downloadReport;
    window.closeAnalysisModal = closeAnalysisModal;
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
