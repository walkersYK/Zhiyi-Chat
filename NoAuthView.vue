<template>
  <div class="container">
    <div class="column one-sixth">
      <ChatLeft />
    </div>
    <div
      class="column two-sixths"
      style="border-left: 4px solid gray; border-right: 4px solid gray"
    >
      <div class="scrollable-content">
        <!-- 这里可以放置大量的内容 -->
        <ChatAi />
        <p>Chat Setting</p>
      </div>
    </div>
    <div class="column three-sixths">
      <div class="grid-container">
        <div class="upper-section">
          <!-- 这里放置 ECharts 渲染的知识图谱 -->
          <div id="chart" style="width: 100%; height: 100%"></div>
        </div>
        <div class="lower-section">
          <div class="table-grid">
            <TableBar />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script lang="ts">
import * as echarts from "echarts";
import ChatAi from "@/components/ChatAi.vue";
import TableBar from "@/components/TableBar.vue";
import ChatLeft from "@/components/ChatLeft.vue";

export default {
  name: "HomeComponent",
  components: {
    // eslint-disable-next-line vue/no-unused-components
    ChatAi,
    // eslint-disable-next-line vue/no-unused-components
    TableBar,
    // eslint-disable-next-line vue/no-unused-components
    ChatLeft, // 注册 ChatAi 组件
  },
  mounted() {
    // 初始化 ECharts 图表
    const chartDom = document.getElementById("chart")!;
    const myChart = echarts.init(chartDom, "dark");
    myChart.showLoading();

    const webkitDep = {
      nodes: [
        { name: "Node 1", category: 0 },
        { name: "Node 2", category: 1 },
        { name: "Node 3", category: 2 },
        { name: "Node 4", category: 3 },
        { name: "Node 5", category: 4 },
      ],
      categories: [
        { name: "HTMLElement" },
        { name: "WebGL" },
        { name: "SVG" },
        { name: "CSS" },
        { name: "Other" },
      ],
      links: [
        { source: 0, target: 1 },
        { source: 1, target: 2 },
        { source: 2, target: 3 },
        { source: 3, target: 4 },
      ],
    };

    myChart.hideLoading();

    const option: echarts.EChartsOption = {
      legend: {
        data: ["HTMLElement", "WebGL", "SVG", "CSS", "Other"],
      },
      series: [
        {
          type: "graph",
          layout: "force",
          animation: false,
          label: {
            position: "right",
            formatter: "{b}",
          },
          draggable: true,
          data: webkitDep.nodes.map((node: any, idx: number) => ({
            ...node,
            id: idx,
          })),
          categories: webkitDep.categories,
          force: {
            edgeLength: 5,
            repulsion: 20,
            gravity: 0.2,
          },
          edges: webkitDep.links,
        },
      ],
    };

    myChart.setOption(option);
  },
};
</script>
<style scoped>
.container {
  display: flex;
  height: 100vh; /* 使容器高度占满整个视口 */
  overflow: hidden; /* 防止内容溢出容器 */
}

.column {
  display: flex;
  flex-direction: column; /* 垂直布局 */
  align-items: center;
  justify-content: center;
  text-align: center;
}

.one-sixth {
  flex: 1; /* 占总宽度的 1/6 */
  background-color: #f4f4f4; /* 可选：背景色 */
  overflow-y: auto; /* 当内容超出高度时，显示滚动条 */
}

.two-sixths {
  flex: 2; /* 占总宽度的 2/6 */
  display: flex;
  background-color: #e0e0e0; /* 可选：背景色 */
  position: relative;
  height: 100%; /* 使中间列高度占满整个容器 */
  overflow-y: auto; /* 当内容超出高度时，显示滚动条 */
}

.scrollable-content {
  height: 100%; /* 设置高度为100%，使其占满整个中间列 */
  padding: 10px; /* 可选：内边距 */
  box-sizing: border-box; /* 使内边距和边框包含在高度内 */
}

/* 隐藏滚动条（Chrome, Safari 和 Opera） */
.scrollable-content::-webkit-scrollbar {
  display: none;
}

.three-sixths {
  flex: 3; /* 占总宽度的 3/6 */
  background-color: #d0d0d0; /* 可选：背景色 */
  display: flex;
  flex-direction: column;
  overflow-y: auto; /* 当内容超出高度时，显示滚动条 */
}

.grid-container {
  height: 100%;
  width: 100%;
  display: flex;
  flex-direction: column;
}

.upper-section {
  flex: 3; /* 占总高度的 3/5 */
  background-color: #c0c0c0; /* 可选：背景色 */
  display: flex;
  align-items: center;
  justify-content: center;
}

.lower-section {
  flex: 2; /* 占总高度的 2/5 */
  display: flex;
  align-items: center;
  justify-content: center;
}

.table-grid {
  width: 100%;
  height: 100%;
  overflow-y: auto; /* 当内容超出高度时，显示滚动条 */
}
</style>
