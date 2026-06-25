import { defineConfig } from "vitepress";
import llmstxt from "vitepress-plugin-llms";

interface TransformPageData {
  title: string;
  description: string;
  relativePath: string;
  lastUpdated: number;
  frontmatter: {
    head?: unknown[];
    description?: string;
    [key: string]: unknown;
  };
}

// Site configuration
export const SITE_URL = "https://muhammad-fiaz.github.io/iris";
export const SITE_NAME = "Iris";
export const SITE_DESCRIPTION =
  "A fast computer vision library framework in Rust. Features include GPU-acceleration, image filters, camera integration, drawing canvas, and native ONNX inference.";

// Google Analytics and Google Tag Manager IDs
export const GA_ID = "G-6BVYCRK57P";
export const GTM_ID = "GTM-P4M9T8ZR";

// Google AdSense Client ID
export const ADSENSE_CLIENT_ID = "ca-pub-2040560600290490";

// SEO Keywords
export const KEYWORDS =
  "rust, computer vision, image processing, video processing, deep learning, burn framework, wgpu, cuda, machine learning, memory safe, type safe, rust-cv, image filtering, edge detection, camera calibration, iris, object detection, face recognition, QR code, optical flow, morphological operations, segmentation, contours, GPU acceleration";

export default defineConfig({
  lang: "en-US",
  title: SITE_NAME,
  description: SITE_DESCRIPTION,
  base: "/iris/",
  lastUpdated: true,
  cleanUrls: true,

  sitemap: {
    hostname: "https://muhammad-fiaz.github.io/iris",
  },

  vite: {
    plugins: [llmstxt()],
  },

  head: [
    [
      "meta",
      { name: "viewport", content: "width=device-width, initial-scale=1.0" },
    ],
    ["meta", { name: "google-adsense-account", content: ADSENSE_CLIENT_ID }],
    // Primary Meta Tags
    ["meta", { name: "title", content: SITE_NAME }],
    ["meta", { name: "description", content: SITE_DESCRIPTION }],
    ["meta", { name: "keywords", content: KEYWORDS }],
    ["meta", { name: "author", content: "Muhammad Fiaz" }],
    ["meta", { name: "robots", content: "index, follow" }],
    ["meta", { name: "language", content: "English" }],
    ["meta", { name: "revisit-after", content: "7 days" }],
    ["meta", { name: "generator", content: "VitePress" }],

    // Open Graph / Facebook
    ["meta", { property: "og:type", content: "website" }],
    ["meta", { property: "og:url", content: SITE_URL }],
    ["meta", { property: "og:title", content: SITE_NAME }],
    ["meta", { property: "og:description", content: SITE_DESCRIPTION }],
    ["meta", { property: "og:image", content: `${SITE_URL}/logo.svg` }],
    ["meta", { property: "og:image:width", content: "1200" }],
    ["meta", { property: "og:image:height", content: "630" }],
    [
      "meta",
      {
        property: "og:image:alt",
        content: "Iris - Rust computer vision and deep learning library",
      },
    ],
    ["meta", { property: "og:site_name", content: SITE_NAME }],
    ["meta", { property: "og:locale", content: "en_US" }],

    // Twitter Card
    ["meta", { name: "twitter:card", content: "summary" }],
    ["meta", { name: "twitter:url", content: SITE_URL }],
    ["meta", { name: "twitter:title", content: SITE_NAME }],
    ["meta", { name: "twitter:description", content: SITE_DESCRIPTION }],
    ["meta", { name: "twitter:image", content: `${SITE_URL}/logo.svg` }],
    ["meta", { name: "twitter:creator", content: "@muhammadfiaz_" }],

    // Favicons
    ["link", { rel: "icon", href: "/iris/logo.svg" }],

    // Theme color
    ["meta", { name: "theme-color", content: "#5e35b1" }],
    ["meta", { name: "msapplication-TileColor", content: "#5e35b1" }],

    // Google Analytics (gtag.js)
    [
      "script",
      {
        async: "",
        src: `https://www.googletagmanager.com/gtag/js?id=${GA_ID}`,
      },
    ],
    [
      "script",
      {},
      `window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', '${GA_ID}');`,
    ],

    // Google Tag Manager
    ...(GTM_ID
      ? ([
          [
            "script",
            {},
            `(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start': new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0], j=d.createElement(s), dl=l!='dataLayer'?'&l='+l:''; j.async=true; j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl; f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','${GTM_ID}');`,
          ],
          [
            "noscript",
            {},
            `<iframe src="https://www.googletagmanager.com/ns.html?id=${GTM_ID}" height="0" width="0" style="display:none;visibility:hidden"></iframe>`,
          ],
        ] as [string, Record<string, string>, string][])
      : []),

    // Google AdSense
    [
      "script",
      {
        async: "",
        src: `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${ADSENSE_CLIENT_ID}`,
        crossorigin: "anonymous",
      },
    ],
  ],

  ignoreDeadLinks: [/.*\.rs$/],

  transformPageData(pageData: TransformPageData) {
    // Dynamic OG image generation based on page title
    const pageTitle = pageData.title || SITE_NAME;
    const pageDescription = pageData.description || SITE_DESCRIPTION;
    const canonicalUrl = `${SITE_URL}/${pageData.relativePath.replace(/((^|\/)index)?\.md$/, "$2").replace(/\.md$/, "")}`;

    pageData.frontmatter.head ??= [];
    pageData.frontmatter.head.push(
      ["link", { rel: "canonical", href: canonicalUrl }],
      [
        "meta",
        { property: "og:title", content: `${pageTitle} | ${SITE_NAME}` },
      ],
      ["meta", { property: "og:url", content: canonicalUrl }],
    );

    if (pageData.frontmatter.description) {
      pageData.frontmatter.head.push(
        [
          "meta",
          {
            property: "og:description",
            content: pageData.frontmatter.description,
          },
        ],
        [
          "meta",
          { name: "description", content: pageData.frontmatter.description },
        ],
      );
    }

    // Dynamic JSON-LD Schema
    const isHome = pageData.relativePath === "index.md";
    const lastUpdated = pageData.lastUpdated
      ? new Date(pageData.lastUpdated).toISOString()
      : new Date().toISOString();

    // Base Graph
    const graph: Record<string, unknown>[] = [];

    // 1. WebSite Schema
    if (isHome) {
      graph.push({
        "@type": "WebSite",
        name: SITE_NAME,
        url: SITE_URL,
        description: SITE_DESCRIPTION,
        author: {
          "@type": "Person",
          name: "Muhammad Fiaz",
          url: "https://github.com/muhammad-fiaz",
        },
      });
    }

    // 2. Main Entity Schema
    const authorSchema = {
      "@type": "Person",
      name: "Muhammad Fiaz",
      url: "https://muhammadfiaz.com",
      sameAs: [
        "https://github.com/muhammad-fiaz",
        "https://www.linkedin.com/in/muhammad-fiaz-",
        "https://x.com/muhammadfiaz_",
      ],
    };

    const primarySchema: Record<string, unknown> = {
      "@type": isHome ? "SoftwareApplication" : "TechArticle",
      name: isHome ? SITE_NAME : pageTitle,
      description: pageDescription,
      url: canonicalUrl,
      image: `${SITE_URL}/logo.svg`,
      author: authorSchema,
      publisher: {
        "@type": "Organization",
        name: SITE_NAME,
        url: SITE_URL,
        logo: {
          "@type": "ImageObject",
          url: `${SITE_URL}/logo.svg`,
        },
      },
    };

    if (isHome) {
      Object.assign(primarySchema, {
        applicationCategory: "DeveloperApplication",
        operatingSystem: "Cross-platform",
        programmingLanguage: "Rust",
        offers: {
          "@type": "Offer",
          price: "0",
          priceCurrency: "USD",
        },
        downloadUrl: "https://github.com/muhammad-fiaz/iris",
        softwareVersion: "0.0.0",
        license: "https://opensource.org/licenses/MIT",
      });
    } else {
      const pathParts: string[] = String(pageData.relativePath).split("/");
      const section =
        pathParts.length > 1
          ? pathParts[0].charAt(0).toUpperCase() + pathParts[0].slice(1)
          : "Documentation";

      Object.assign(primarySchema, {
        headline: pageTitle,
        articleSection: section,
        mainEntityOfPage: {
          "@type": "WebPage",
          "@id": canonicalUrl,
        },
        datePublished: "2026-06-24T00:00:00Z",
        dateModified: lastUpdated,
      });
    }
    graph.push(primarySchema);

    // 3. BreadcrumbList Schema
    const breadcrumbs: Record<string, unknown>[] = [
      {
        "@type": "ListItem",
        position: 1,
        name: "Home",
        item: SITE_URL,
      },
    ];

    if (!isHome) {
      const pathParts: string[] = String(pageData.relativePath)
        .replace(/\.md$/, "")
        .split("/");
      let currentPath = SITE_URL;

      pathParts.forEach((part: string, index: number) => {
        currentPath += `/${part}`;
        const name = part
          .split("-")
          .map((s: string) => s.charAt(0).toUpperCase() + s.slice(1))
          .join(" ");

        breadcrumbs.push({
          "@type": "ListItem",
          position: index + 2,
          name: name,
          item: index === pathParts.length - 1 ? canonicalUrl : currentPath,
        });
      });
    }

    graph.push({
      "@type": "BreadcrumbList",
      itemListElement: breadcrumbs,
    });

    pageData.frontmatter.head.push([
      "script",
      { type: "application/ld+json" },
      JSON.stringify({
        "@context": "https://schema.org",
        "@graph": graph,
      }),
    ]);
  },

  themeConfig: {
    logo: "/logo.svg",
    siteTitle: "Iris",

    nav: [
      { text: "Home", link: "/" },
      { text: "Guide", link: "/guide/getting-started" },
      { text: "API", link: "/api/" },
      { text: "Examples", link: "/guide/examples" },
      {
        text: "Support",
        items: [
          {
            text: "💖 Sponsor",
            link: "https://github.com/sponsors/muhammad-fiaz",
          },
          { text: "☕ Donate", link: "https://pay.muhammadfiaz.com" },
        ],
      },
      { text: "GitHub", link: "https://github.com/muhammad-fiaz/iris" },
    ],

    sidebar: {
      "/guide/": [
        {
          text: "Introduction",
          items: [
            { text: "What is Iris?", link: "/guide/" },
            { text: "Introduction", link: "/guide/introduction" },
            { text: "Getting Started", link: "/guide/getting-started" },
            { text: "Installation", link: "/guide/installation" },
          ],
        },
        {
          text: "Core Features",
          items: [
            { text: "Image Representation", link: "/guide/image" },
            { text: "Image Filters & Blur", link: "/guide/filters" },
            { text: "Edge & Gradients", link: "/guide/edges" },
            { text: "Morphology", link: "/guide/morphology" },
            { text: "Drawing & Canvas", link: "/guide/drawing" },
          ],
        },
        {
          text: "Advanced Capabilities",
          items: [
            { text: "DNN & ONNX Inference", link: "/guide/dnn" },
            { text: "ArUco & QR Detection", link: "/guide/detection" },
            { text: "Segmentation & Contours", link: "/guide/contours" },
            { text: "Tracking & Flow", link: "/guide/tracking" },
            { text: "Camera & Calibration", link: "/guide/camera" },
            { text: "Examples", link: "/guide/examples" },
          ],
        },
      ],
      "/api/": [
        {
          text: "API Reference",
          items: [
            { text: "Overview", link: "/api/" },
            { text: "Core Module", link: "/api/core" },
            { text: "Image Operators", link: "/api/image" },
            { text: "Filters", link: "/api/filters" },
            { text: "Color", link: "/api/color" },
            { text: "Edges", link: "/api/edges" },
            { text: "Morphology", link: "/api/morphology" },
            { text: "Threshold", link: "/api/threshold" },
            { text: "Histogram", link: "/api/histogram" },
            { text: "Drawing", link: "/api/drawing" },
            { text: "Noise", link: "/api/noise" },
            { text: "Contours", link: "/api/contours" },
            { text: "Features", link: "/api/features" },
            { text: "Tracking", link: "/api/tracking" },
            { text: "DNN Module", link: "/api/dnn" },
            { text: "Video", link: "/api/video" },
            { text: "Inpainting", link: "/api/inpaint" },
            { text: "Stereo Vision", link: "/api/stereo" },
            { text: "Kalman Filter", link: "/api/kalman" },
            { text: "HOG Descriptor", link: "/api/hog" },
            { text: "Photo Processing", link: "/api/photo" },
            { text: "Image Stitching", link: "/api/stitching" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/muhammad-fiaz/iris" },
    ],

    footer: {
      message: "Released under the MIT License.",
      copyright: `Copyright © 2026-${new Date().getFullYear()} Muhammad Fiaz`,
    },

    search: {
      provider: "local",
    },

    editLink: {
      pattern:
        "https://github.com/muhammad-fiaz/iris/edit/main/docs/:path",
      text: "Edit this page on GitHub",
    },

    lastUpdated: {
      text: "Last updated",
      formatOptions: {
        dateStyle: "medium",
        timeStyle: "short",
      },
    },
  },
});
