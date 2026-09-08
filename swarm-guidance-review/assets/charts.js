(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });

  var el = document.getElementById('chart-sample');
  if (el) {
    var chart = echarts.init(el, null, { renderer: 'svg' });
    chart.setOption({
      animation: false,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, appendToBody: true },
      grid: { left: 16, right: 24, top: 36, bottom: 10, containLabel: true },
      xAxis: {
        type: 'value',
        minInterval: 1,
        axisLabel: { color: muted, fontSize: 12 },
        splitLine: { lineStyle: { color: rule } },
        axisLine: { show: false }
      },
      yAxis: {
        type: 'category',
        inverse: true,
        data: ['基础综述', '体系博弈/任务分配', '单机突防-打击一体化', 'MARL 集群突防与规避', '博弈对抗协同制导', '时空协同打击制导'],
        axisLabel: { color: ink, fontSize: 12.5, fontWeight: 600 },
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false }
      },
      series: [{
        type: 'bar',
        barWidth: '55%',
        data: [1, 5, 6, 7, 8, 3],
        label: { show: true, position: 'right', color: ink, fontSize: 12, fontWeight: 700 },
        itemStyle: {
          color: function(params) {
            return params.dataIndex === 2 ? accent2 : accent;
          },
          borderRadius: [0, 3, 3, 0]
        }
      }]
    });
    window.addEventListener('resize', function() { chart.resize(); });
  }
})();
