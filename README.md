# willburnstech

A technical blog built with [Jekyll](https://jekyllrb.com/) and the [Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy), hosted on GitHub Pages.

**Live Site:** [https://willburnstech.github.io](https://willburnstech.github.io)

## About

This blog covers topics in:
- AI/Automation & LLM Engineering
- Penetration Testing & Security Research
- Full-Stack Development

## Local Development

### Prerequisites

- [Ruby](https://www.ruby-lang.org/en/downloads/) (version 3.1 or higher)
- [Bundler](https://bundler.io/)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/willburnstech/willburnstech.github.io.git
   cd willburnstech.github.io
   ```

2. **Install dependencies:**
   ```bash
   bundle install
   ```

3. **Start the local server:**
   ```bash
   bundle exec jekyll serve
   ```

4. **View the site:**
   Open [http://127.0.0.1:4000](http://127.0.0.1:4000) in your browser.

### Live Reload (Optional)

For automatic browser refresh on changes:
```bash
bundle exec jekyll serve --livereload
```

## Writing Posts

Create new posts in the `_posts/` directory following the naming convention:
```
YYYY-MM-DD-title-of-post.md
```

### Front Matter Template

```yaml
---
title: "Your Post Title"
date: YYYY-MM-DD HH:MM:SS +1000
categories: [Category1, Category2]
tags: [tag1, tag2, tag3]
description: "Brief description for SEO"
image:
  path: /assets/img/posts/image.png
  alt: "Image description"
pin: false        # Pin to top of home page
math: false       # Enable MathJax
mermaid: false    # Enable Mermaid diagrams
---

Your content here...
```

### Content Features

- **Code blocks:** Use triple backticks with language identifier
- **Images:** Place in `/assets/img/posts/` and reference with relative paths
- **Prompts:** Use `{: .prompt-tip }`, `{: .prompt-info }`, `{: .prompt-warning }`, `{: .prompt-danger }`

## Project Structure

```
.
├── _config.yml          # Site configuration
├── _data/
│   └── contact.yml      # Footer contact links
├── _posts/              # Blog posts
├── _tabs/
│   └── about.md         # About page
├── assets/
│   └── img/             # Images and media
├── .github/
│   └── workflows/       # GitHub Actions deployment
├── Gemfile              # Ruby dependencies
└── index.html           # Home page
```

## Deployment

The site automatically deploys to GitHub Pages when you push to the `main` branch.

### Manual Deployment

The GitHub Actions workflow (`.github/workflows/pages-deploy.yml`) handles:
1. Building the Jekyll site
2. Running HTML validation
3. Deploying to GitHub Pages

### Initial GitHub Pages Setup

1. Go to your repository Settings > Pages
2. Under "Build and deployment", select **GitHub Actions** as the source
3. Push to `main` branch to trigger the first deployment

## Customization

### Update Site Info
Edit `_config.yml` to update:
- Site title and tagline
- Social links
- Avatar image
- Google Analytics ID
- Comments provider

### Add Avatar
1. Add your avatar image to `/assets/img/avatar.png`
2. Update the `avatar` path in `_config.yml` if using a different filename

### Enable Comments
Configure Giscus, Disqus, or Utterances in `_config.yml` under the `comments` section.

## Resources

- [Chirpy Theme Documentation](https://chirpy.cotes.page/)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

## License

Content is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Code snippets are available under the [MIT License](LICENSE).

---

**Author:** Will Burns
**Contact:** [willburns.tech](https://willburns.tech) | [LinkedIn](https://linkedin.com/in/willburnstech) | [GitHub](https://github.com/willburnstech)
