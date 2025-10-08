$(document).ready(function () {
  // Lógica para el formulario de usuario
  $("#btnUsuario").on("click", function () {
    $("#formularioUsuario").fadeIn(300);
  });

  $("#btnCerrar").on("click", function () {
    $("#formularioUsuario").fadeOut(300);
  });

  $("#btnCancelar").on("click", function () {
    $("#formularioUsuario").fadeOut(300);
  });

  $("#btnGuardar").on("click", function () {
    const usuarioSeleccionado = $("#nombre option:selected").text();
    console.log("Usuario seleccionado:", usuarioSeleccionado);
    
    // Mostrar mensaje de éxito
    const statusIndicator = document.getElementById('statusIndicator');
    if (statusIndicator) {
      const statusText = statusIndicator.querySelector('.status-text');
      if (statusText) {
        statusText.textContent = `CONECTADO // ${usuarioSeleccionado}`;
      }
    }
    
    $("#formularioUsuario").fadeOut(300);
  });

  // Cerrar formulario al hacer clic fuera de él
  $(document).on("click", function (event) {
    if ($(event.target).is("#formularioUsuario")) {
      $("#formularioUsuario").fadeOut(300);
    }
  });

  // Efectos de hover para las tarjetas de películas
  $(document).on("mouseenter", ".movie-card", function () {
    $(this).addClass("hover");
  });

  $(document).on("mouseleave", ".movie-card", function () {
    $(this).removeClass("hover");
  });

  // Animación de aparición para elementos
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px"
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = "1";
        entry.target.style.transform = "translateY(0)";
      }
    });
  }, observerOptions);

  // Observar elementos que aparecen
  $(".movie-card, .cyber-panel, .section-header").each(function () {
    this.style.opacity = "0";
    this.style.transform = "translateY(20px)";
    this.style.transition = "opacity 0.5s ease, transform 0.5s ease";
    observer.observe(this);
  });

  // Efecto de glitch aleatorio en el logo
  setInterval(() => {
    const glitchElements = document.querySelectorAll('.glitch');
    glitchElements.forEach(el => {
      if (Math.random() > 0.95) {
        el.style.animation = 'none';
        setTimeout(() => {
          el.style.animation = '';
        }, 50);
      }
    });
  }, 3000);

  // Efecto de escaneo en paneles
  $(".cyber-panel").each(function () {
    if (!$(this).find('.scan-line').length) {
      $(this).append('<div class="scan-line"></div>');
    }
  });

  console.log("CYBER FLIX // Sistema inicializado");
});
