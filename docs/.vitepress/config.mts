import { withMermaid } from "vitepress-plugin-mermaid"
import mathjax3 from 'markdown-it-mathjax3'
// import { defineConfig } from 'vitepress'

// 通过 JavaScript 动态处理侧边栏配置：
function processSidebar(sidebarConfig) {
  if (Array.isArray(sidebarConfig)) {
    return sidebarConfig.map(group => ({
      ...group,
      collapsible: true,
      collapsed: true,
      items: group.items?.map(item => ({
        ...item,
        ...(item.items && { collapsible: true, collapsed: true })
      }))
    }))
  } else if (typeof sidebarConfig === 'object') {
    return Object.fromEntries(
      Object.entries(sidebarConfig).map(([path, groups]) => [
        path,
        processSidebar(groups) // 递归处理每个路径的侧边栏
      ])
    )
  }
  return sidebarConfig
}


const rawSidebar = {
  '/python/basic/': [
    {
      text: 'Python基础',
      items: [
        { text: '前言', link: '/python/basic/' },
        { text: '环境安装指南', link: '/python/basic/installation' },
        { text: '基础语法', link: '/python/basic/syntax' },
        { text: '常用数据结构', link: '/python/basic/data_structures' },
        { text: '函数详解', link: '/python/basic/function' },
        { text: '面向对象编程', link: '/python/basic/oop' },
        { text: '异常处理', link: '/python/basic/exceptions' },
        { text: '文件操作', link: '/python/basic/files' },
        { text: '模块与包管理', link: '/python/basic/modules_packages' },
        { text: '正则表达式', link: '/python/basic/re' },
        { text: '多线程与多进程', link: '/python/basic/multiprocess_thread' },
        { text: '全局解释器锁', link: '/python/basic/gil' },
        { text: '异步编程', link: '/python/basic/asyncio' },
        { text: '数据库操作', link: '/python/basic/sql' },
        { text: 'Web开发入门', link: '/python/basic/web' }
      ]
    }
  ],
  '/python/data_structure/': [
    {
      text: '数据结构与算法',
      items: [
        { text: '前言', link: '/python/data_structure/' },
        { text: '初识算法', link: '/python/data_structure/first_meet' },
        { text: '算法复杂度', link: '/python/data_structure/complexity' },
        { text: '数据结构', link: '/python/data_structure/data_structure' },
        {
          text: '数组与链表', items: [
            { text: '数组', link: '/python/data_structure/array' },
            { text: '链表', link: '/python/data_structure/linked_list' },
            { text: '列表', link: '/python/data_structure/list' },
            { text: '内存与缓存', link: '/python/data_structure/ram_cache' },
            { text: '小结', link: '/python/data_structure/summary4' }
          ]
        },
        {
          text: '栈与队列', items: [
            { text: '栈', link: '/python/data_structure/stack' },
            { text: '队列', link: '/python/data_structure/queue' },
            { text: '双向队列', link: '/python/data_structure/bidirectional_queue' },
            { text: '小结', link: '/python/data_structure/summary5' }
          ]
        },
        {
          text: '哈希表', items: [
            { text: '哈希表', link: '/python/data_structure/hashtable' },
            { text: '哈希冲突', link: '/python/data_structure/hash_collisions' },
            { text: '哈希算法', link: '/python/data_structure/hash_algorithm' },
            { text: '小结', link: '/python/data_structure/summary6' }
          ]
        },
        {
          text: '树', items: [
            { text: '二叉树', link: '/python/data_structure/binary_tree' },
            { text: '二叉树遍历', link: '/python/data_structure/binary_tree_traversal' },
            { text: '二叉树数组表示', link: '/python/data_structure/binary_tree_array' },
            { text: 'AVL树', link: '/python/data_structure/avl_tree' },
            { text: '小结', link: '/python/data_structure/summary7' }
          ]
        },
        {
          text: '堆', items: [
            { text: '堆', link: '/python/data_structure/heap' },
            { text: '建堆问题', link: '/python/data_structure/heap_building' },
            { text: 'Tok-k问题', link: '/python/data_structure/top_k' },
            { text: '小结', link: '/python/data_structure/summary8' }
          ]
        },
        {
          text: '图', items: [
            { text: '图', link: '/python/data_structure/graph' },
            { text: '图基础操作', link: '/python/data_structure/graph_operate' },
            { text: '图的遍历', link: '/python/data_structure/graph_traversal' },
            { text: '小结', link: '/python/data_structure/summary9' }
          ]
        },
        {
          text: '搜索', items: [
            { text: '二分查找', link: '/python/data_structure/binary_search' },
            { text: '二分查找插入点', link: '/python/data_structure/binary_search_insertion_point' },
            { text: '二分查找边界', link: '/python/data_structure/binary_search_boundary' },
            { text: '哈希优化策略', link: '/python/data_structure/hash_optimization_strategy' },
            { text: '重识搜索算法', link: '/python/data_structure/rerecognition_search_algorithm' },
            { text: '小结', link: '/python/data_structure/summary10' }
          ]
        },
        {
          text: '排序', items: [
            { text: '排序算法', link: '/python/data_structure/binary_search' },
            { text: '选择排序', link: '/python/data_structure/binary_search_insertion_point' },
            { text: '冒泡排序', link: '/python/data_structure/binary_search_boundary' },
            { text: '插入排序', link: '/python/data_structure/hash_optimization_strategy' },
            { text: '快速排序', link: '/python/data_structure/rerecognition_search_algorithm' },
            { text: '归并排序', link: '/python/data_structure/rerecognition_search_algorithm' },
            { text: '堆排序', link: '/python/data_structure/rerecognition_search_algorithm' },
            { text: '桶排序', link: '/python/data_structure/rerecognition_search_algorithm' },
            { text: '计数排序', link: '/python/data_structure/rerecognition_search_algorithm' },
            { text: '基数排序', link: '/python/data_structure/rerecognition_search_algorithm' },
            { text: '小结', link: '/python/data_structure/summary11' }
          ]
        },
        {
          text: '分治', items: [
            { text: '分治算法', link: '/python/data_structure/heap' },
            { text: '分治搜索策略', link: '/python/data_structure/heap_building' },
            { text: '构建树问题', link: '/python/data_structure/top_k' },
            { text: '汉诺塔问题', link: '/python/data_structure/top_k' },
            { text: '小结', link: '/python/data_structure/summary12' }
          ]
        },
        {
          text: '回溯', items: [
            { text: '回溯算法', link: '/python/data_structure/heap' },
            { text: '全排列问题', link: '/python/data_structure/heap_building' },
            { text: '子集和问题', link: '/python/data_structure/top_k' },
            { text: 'N皇后问题', link: '/python/data_structure/top_k' },
            { text: '小结', link: '/python/data_structure/summary13' }
          ]
        },
        {
          text: '动态规划', items: [
            { text: '初探动态规划', link: '/python/data_structure/heap' },
            { text: 'DP问题特性', link: '/python/data_structure/heap_building' },
            { text: 'DP解题思路', link: '/python/data_structure/top_k' },
            { text: '0-1背包问题', link: '/python/data_structure/top_k' },
            { text: '完全背包问题', link: '/python/data_structure/top_k' },
            { text: '编辑距离问题', link: '/python/data_structure/top_k' },
            { text: '小结', link: '/python/data_structure/summary14' }
          ]
        },
        {
          text: '贪心', items: [
            { text: '贪心算法', link: '/python/data_structure/heap' },
            { text: '分数背包问题', link: '/python/data_structure/heap_building' },
            { text: '最大容量问题', link: '/python/data_structure/top_k' },
            { text: '最大切分乘积问题', link: '/python/data_structure/top_k' },
            { text: '小结', link: '/python/data_structure/summary15' }
          ]
        },
      ]
    }
  ],
  '/python/backend/': [
    {
      text: '后端',
      items: [
        { text: 'flask', link: '/python_basics/installation' },
        { text: 'django', link: '/python_basics/syntax' },
      ]
    }
  ],
  '/python/spider/': [
    {
      text: '爬虫',
      items: [
        { text: 'requests', link: '/python_basics/installation' },
        { text: 'bs4', link: '/python_basics/syntax' },
        { text: 'scrapy', link: '/python_basics/oop' }
      ]
    }
  ],
  '/ai/ml/': [
    {
      text: '机器学习',
      items: [
        { text: '数学基础', link: '/ml/math' },
        { text: '线性回归', link: '/ml/linear_regression' },
        { text: '分类算法', link: '/ml/classification' }
      ]
    }
  ],
  '/ai/llm/': [
    {
      text: '大模型技术',
      items: [
        { text: 'Transformer', link: '/llm/transformer' },
        { text: '预训练实践', link: '/llm/pretrain' },
        { text: '微调技巧', link: '/llm/finetune' }
      ]
    }
  ]
}



export default withMermaid({
  title: "智学笔记",
  description: "编程与人工智能学习笔记",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    outline: [2, 3, 4],
    nav: [
      { text: '首页', link: '/' },
      {
        text: 'Python',
        items: [
          { text: 'Python基础', link: '/python/basic/' },
          { text: '算法与数据结构', link: '/python/data_structure/' },
          { text: '后端', link: '/python/backend/' },
          { text: '爬虫', link: '/python/spider/' },
        ]
      },
      {
        text: '人工智能',
        items: [
          { text: '机器学习', link: '/ai/ml/' },
          { text: '深度学习', link: '/ai/dl/' },
          { text: '大模型', link: '/ai/llm/' }
        ]
      },
      {
        text: '统计学', items: [
          { text: '概率论', link: '/statistics/probability/' },
          { text: '微积分', link: '/statistics/calculus/' }
        ]
      }
    ],

    sidebar: processSidebar(rawSidebar),

    socialLinks: [
      { icon: 'github', link: 'https://github.com/chenxilin1994' }
    ],

    editLink: {
      pattern: 'https://github.com/yourusername/yourrepo/edit/main/:path',
      text: '在GitHub上编辑此页'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档'
          }
        }
      }
    }
  },

  markdown: {
    lineNumbers: true,
    theme: 'material-theme-palenight',
    config: (md) => {
      md.use(mathjax3)
    }
  }
})