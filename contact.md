---
layout: page
title: Contact
permalink: /contact/
---

Fill in the form or [email me](mailto:{{site.email}}) to discuss your next project.

{% include form.html %}

<form
  action="https://formspree.io/f/xdopybbw"
  method="POST"
>
  <label>
    Your email:
    <input type="text" name="_replyto">
  </label>
  <label>
    Your message:
    <textarea name="message"></textarea>
  </label>

  <button type="submit">Send</button>
</form>
{% include modal.html %}
