import { withMermaid } from "vitepress-plugin-mermaid"
import mathjax3 from 'markdown-it-mathjax3'
// import { defineConfig } from 'vitepress'

export default withMermaid({
  title: "智学笔记",
  description: "编程与人工智能学习笔记",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config

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

    sidebar: {
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
            { text: '数组', link: '/python/data_structure/array' },
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
    },

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