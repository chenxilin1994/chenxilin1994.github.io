import { withMermaid } from "vitepress-plugin-mermaid"
import mathjax3 from 'markdown-it-mathjax3'
import { container } from '@mdit/plugin-container'

// 通过 JavaScript 动态处理侧边栏配置：
function processSidebar(sidebarConfig) {
  if (Array.isArray(sidebarConfig)) {
    return sidebarConfig.map(group => ({
      ...group,
      collapsible: true,
      collapsed: false,
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
  '/python/basic': [
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
        { text: 'Web开发入门', link: '/python/basic/web' },
        {
          text: '高级编程', items: [
            { text: '一切皆对象', link: '/python/basic/advanced/everything_is_object' },
            { text: '魔法函数', link: '/python/basic/advanced/magic_function' },
            { text: '深入类和对象', link: '/python/basic/advanced/more_about_class_object' }
          ]
        },
      ]
    }],
  '/python/interview': [
    { text: '基础语法', link: '/python/interview/basic_syntax' },
    { text: '数据结构', link: '/python/interview/data_structure' },
    { text: '函数与模块', link: '/python/interview/function_module' },
    { text: '面向对象编程', link: '/python/interview/oop' },
    { text: '并发与异步编程', link: '/python/interview/concurrency_asyncio' },
    { text: '模块与包管理', link: '/python/interview/module_package' },
    { text: '错误处理与调试', link: '/python/interview/exception_debug' },
    { text: '数据库与ORM', link: '/python/interview/database_orm' },
    { text: '测试', link: '/python/interview/test' },
    { text: '高级编程', link: '/python/interview/advanced' },
  ],

  '/python/data_structure': [
    {
      text: '数据结构与算法',
      items: [
        { text: '前言', link: '/python/data_structure/' },
        { text: '初识算法', link: '/python/data_structure/first_meet' },
        {
          text: '复杂度分析',
          items: [
            { text: '算法效率评估', link: '/python/data_structure/complexity_analysis/algorithm_efficiency_evaluate' },
            { text: '迭代与递归', link: '/python/data_structure/complexity_analysis/iteration_recursion' },
            { text: '时间复杂度', link: '/python/data_structure/complexity_analysis/time_complexity' },
            { text: '空间复杂度', link: '/python/data_structure/complexity_analysis/space_complexity' }
            , { text: '小结', link: '/python/data_structure/complexity_analysis/summary' }
          ]
        },
        {
          text: '数据结构', items: [
            { text: '数据结构分类', link: '/python/data_structure/data_structure/data_structure_classify' },
            { text: '基本数据类型', link: '/python/data_structure/data_structure/basic_data_type' },
            { text: '数字编码', link: '/python/data_structure/data_structure/digitally_encoding' },
            { text: '字符编码', link: '/python/data_structure/data_structure/character_encoding' },
            { text: '小结', link: '/python/data_structure/data_structure/summary' },
          ]
        },
        {
          text: '数组与链表', items: [
            { text: '数组', link: '/python/data_structure/array_link_list/array' },
            { text: '链表', link: '/python/data_structure/array_link_list/linked_list' },
            { text: '列表', link: '/python/data_structure/array_link_list/list' },
            { text: '内存与缓存', link: '/python/data_structure/array_link_list/ram_cache' },
            { text: '小结', link: '/python/data_structure/array_link_list/summary' }
          ]
        },
        {
          text: '栈与队列', items: [
            { text: '栈', link: '/python/data_structure/stack_queue/stack' },
            { text: '队列', link: '/python/data_structure/stack_queue/queue' },
            { text: '双向队列', link: '/python/data_structure/stack_queue/deque' },
            { text: '小结', link: '/python/data_structure/stack_queue/summary' }
          ]
        },
        {
          text: '哈希表', items: [
            { text: '哈希表', link: '/python/data_structure/hash_table/hash_table' },
            { text: '哈希冲突', link: '/python/data_structure/hash_table/hash_collisions' },
            { text: '哈希算法', link: '/python/data_structure/hash_table/hash_algorithm' },
            { text: '小结', link: '/python/data_structure/hash_table/summary' }
          ]
        },
        {
          text: '树', items: [
            { text: '二叉树', link: '/python/data_structure/tree/binary_tree' },
            { text: '二叉树遍历', link: '/python/data_structure/tree/binary_tree_traversal' },
            { text: '二叉树数组表示', link: '/python/data_structure/tree/binary_tree_array' },
            { text: 'AVL树', link: '/python/data_structure/tree/avl_tree' },
            { text: '小结', link: '/python/data_structure/tree/summary' }
          ]
        },
        {
          text: '堆', items: [
            { text: '堆', link: '/python/data_structure/heap/heap' },
            { text: '建堆问题', link: '/python/data_structure/heap/heap_building' },
            { text: 'Tok-k问题', link: '/python/data_structure/heap/top_k' },
            { text: '小结', link: '/python/data_structure/heap/summary' }
          ]
        },
        {
          text: '图', items: [
            { text: '图', link: '/python/data_structure/graph/graph' },
            { text: '图基础操作', link: '/python/data_structure/graph/graph_operate' },
            { text: '图的遍历', link: '/python/data_structure/graph/graph_traversal' },
            { text: '小结', link: '/python/data_structure/graph/summary' }
          ]
        },
        {
          text: '搜索', items: [
            { text: '二分查找', link: '/python/data_structure/search/binary_search' },
            { text: '二分查找插入点', link: '/python/data_structure/search/binary_search_insertion_point' },
            { text: '二分查找边界', link: '/python/data_structure/search/binary_search_boundary' },
            { text: '哈希优化策略', link: '/python/data_structure/search/hash_optimization_strategy' },
            { text: '重识搜索算法', link: '/python/data_structure/search/rerecognition_search_algorithm' },
            { text: '小结', link: '/python/data_structure/search/summary' }
          ]
        },
        {
          text: '排序', items: [
            { text: '排序算法', link: '/python/data_structure/sorting/sorting_algorithm' },
            { text: '选择排序', link: '/python/data_structure/sorting/selection_sort' },
            { text: '冒泡排序', link: '/python/data_structure/sorting/bubble_sort' },
            { text: '插入排序', link: '/python/data_structure/sorting/insertion_sort' },
            { text: '快速排序', link: '/python/data_structure/sorting/quick_sort' },
            { text: '归并排序', link: '/python/data_structure/sorting/merge_sort' },
            { text: '堆排序', link: '/python/data_structure/sorting/heap_sort' },
            { text: '桶排序', link: '/python/data_structure/sorting/bucket_sort' },
            { text: '计数排序', link: '/python/data_structure/sorting/counting_sort' },
            { text: '基数排序', link: '/python/data_structure/sorting/radix_sort' },
            { text: '小结', link: '/python/data_structure/sorting/summary' }
          ]
        },
        {
          text: '分治', items: [
            { text: '分治算法', link: '/python/data_structure/divide_conquer/divide_conquer_algorithm' },
            { text: '分治搜索策略', link: '/python/data_structure/divide_conquer/divide_conquer_search_stragety' },
            { text: '构建树问题', link: '/python/data_structure/divide_conquer/build_binary_tree_problem' },
            { text: '汉诺塔问题', link: '/python/data_structure/divide_conquer/hanota_problem' },
            { text: '小结', link: '/python/data_structure/divide_conquer/summary' }
          ]
        },
        {
          text: '回溯', items: [
            { text: '回溯算法', link: '/python/data_structure/back_tracking/back_tracking_algorithm' },
            { text: '全排列问题', link: '/python/data_structure/back_tracking/permutations_problem' },
            { text: '子集和问题', link: '/python/data_structure/back_tracking/subset_sum_problem' },
            { text: 'N皇后问题', link: '/python/data_structure/back_tracking/n_queens_problem' },
            { text: '小结', link: '/python/data_structure/back_tracking/summary' }
          ]
        },
        {
          text: '动态规划', items: [
            { text: '初探动态规划', link: '/python/data_structure/dynamic_programming/intro_to_dynamic_programming' },
            { text: 'DP问题特性', link: '/python/data_structure/dynamic_programming/dp_problem_features' },
            { text: 'DP解题思路', link: '/python/data_structure/dynamic_programming/dp_solution_pipeline' },
            { text: '0-1背包问题', link: '/python/data_structure/dynamic_programming/knapsack_problem' },
            { text: '完全背包问题', link: '/python/data_structure/dynamic_programming/unbounded_knapsack_problem' },
            { text: '编辑距离问题', link: '/python/data_structure/dynamic_programming/edit_distance_problem' },
            { text: '小结', link: '/python/data_structure/dynamic_programming/summary' }
          ]
        },
        {
          text: '贪心', items: [
            { text: '贪心算法', link: '/python/data_structure/greedy/greedy_algorithm' },
            { text: '分数背包问题', link: '/python/data_structure/greedy/fractional_knapsack_problem' },
            { text: '最大容量问题', link: '/python/data_structure/greedy/max_capacity_problem' },
            { text: '最大切分乘积问题', link: '/python/data_structure/greedy/max_product_cutting_problem' },
            { text: '小结', link: '/python/data_structure/greedy/summary' }
          ]
        },
      ]
    },
  ],
  '/python/backend': [{
    text: '后端开发',
    items: [
      { text: 'flask', link: '/python/baskend/flask' },
      { text: 'django', link: '/python/backend/django' },
    ]
  }],
  '/python/spider': [{
    text: '爬虫技术',
    items: [
      { text: 'requests', link: '/python/spider/requests' },
      { text: 'bs4', link: '/python/spider/bs4' },
      { text: 'scrapy', link: '/python/spider/scrapy' }
    ]
  }],


  '/ai/ml': [
    {
      text: '机器学习',
      items: [
        {
          text: '特征工程', items: [
            { text: '缺失值处理', link: '' },
            { text: '异常值处理', link: '' },
            { text: 'TF-IDF', link: '/ai/ml/feature_engineer/tf-idf' },
          ]
        },
        {
          text: '线性模型', items: [
            { text: '线性回归', link: '' },
            { text: '逻辑回归', link: '' },
            { text: '感知机', link: '' },
          ]
        },
        { text: 'K近邻', link: '' },
        {
          text: '支持向量机', items: [
            { text: '核函数', link: '' },
            { text: '软间隔与硬间隔分类', link: '' },
            { text: 'SVM回归', link: '' },
          ]
        },
        {
          text: '树模型', items: [
            { text: '决策树-ID3', link: '' },
            { text: '决策树-C4.5', link: '' },
            { text: '决策树-CART', link: '' },
            { text: '装袋', link: '' },
            { text: '提升', link: '' },
            { text: 'Adaboost', link: '' },
            { text: '随机森林', link: '' },
            { text: 'GBDT', link: '' },
            { text: 'XGBoost', link: '' },
            { text: 'LightGBM', link: '' },
            { text: 'Catboost', link: '' },
            { text: '感知机', link: '' },
          ]
        },
        {
          text: '贝叶斯', items: [
            { text: '朴素贝叶斯', link: '' },
            { text: '贝叶斯网络', link: '' },
            { text: '贝叶斯线性回归', link: '' },
            { text: '贝叶斯逻辑回归', link: '' }
          ]
        },
        {
          text: '聚类', items: [
            { text: 'K均值聚类', link: '' },
            { text: '层次聚类', link: '' },
            { text: 'DBSCAN', link: '' },
            { text: '高斯混合模型', link: '' },
            { text: 'EM算法', link: '' }
          ]
        },
        {
          text: '降维', items: [
            { text: '主成分分析', link: '' },
            { text: 't-SNE与UMAP', link: '' },
            { text: 'LDA降维', link: '' }
          ]
        },
        {
          text: '关联规则', items: [
            { text: 'Apriori算法', link: '' },
            { text: 'FP-Growth', link: '' }
          ]
        }
      ]
    }
  ],

  '/ai/dl': [
    {
      text: '深度学习',
      items: [
        { text: '基础', link: '/ml/math' },
        { text: '反向传播算法', link: '/ml/linear_regression' },
        { text: '卷积神经网络', link: '/ml/classification' }
      ]
    },

  ],
  '/ai/llm': [
    {
      text: '大模型技术',
      items: [
        { text: 'Transformer', link: '/llm/transformer' },
        { text: '预训练实践', link: '/llm/pretrain' },
        { text: '微调技巧', link: '/llm/finetune' }
      ]
    }
  ],

  '/statistics/probability': [
    {
      text: '概率论', link: '', items: []
    },

  ],
  '/statistics/calculus': [
    {
      text: '微积分', link: '', items: []
    },
  ]
}


export default withMermaid({
  title: "智学笔记",
  description: "编程与人工智能学习笔记",
  themeConfig: {
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
          { text: 'Python面试题', link: '/python/interview/' },
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
      pattern: 'https://github.com/chenxilin1994/chenxilin1994.github.io/edit/master/docs/:path',
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
      md.use(container, {
        name: 'question',
        render: (tokens, idx) => {
          const token = tokens[idx]
          const title = token.info.trim().slice('question'.length).trim()
          return token.nesting === 1
            ? `<div class="question-block">${title ? `<p>${title}</p>` : ''}`
            : '</div>'
        }
      })
    }
  }
})