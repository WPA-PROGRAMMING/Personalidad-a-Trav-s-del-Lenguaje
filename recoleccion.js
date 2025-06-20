// Este archivo sera usado por medio de la herramienta de google https://script.google.com

function crearFormularioPersonalizado() {
  // Define el título del formulario
  var formTitle = 'Encuesta de Frases Incompletas y Datos Demográficos';
  
  // Crea un nuevo formulario
  var form = FormApp.create(formTitle);
  form.setTitle(formTitle)
      .setDescription('Por favor, completa las siguientes preguntas y frases para una mejor comprensión.'); // Puedes cambiar la descripción

  // --- Preguntas de Datos Demográficos ---
  form.addTextItem()
      .setTitle('Edad')
      .setRequired(true); // Hace que la pregunta sea obligatoria

  form.addTextItem()
      .setTitle('Ocupación')
      .setRequired(true);

  form.addTextItem()
      .setTitle('Estado de residencia')
      .setRequired(true);

  // --- Preguntas de Frases Incompletas ---
  var frasesIncompletas = [
    "Siento que mi padre rara vez",
    "Cuando la suerte está en mi contra",
    "Siempre he querido que",
    "Si estuviera encargado",
    "El futuro me parece",
    "Las personas que son mis superiores",
    "Tal vez sea tontería, pero temo",
    "Creo que un verdadero amigo",
    "Cuando era niño",
    "Mi idea de una mujer perfecta",
    "Si veo a una mujer y a un hombre juntos",
    "Comparada con la mayoría de las familias, la mía",
    "En el trabajo me llevo muy bien",
    "Mi madre",
    "Haría cualquier cosa por olvidar la vez que",
    "Si mi padre solamente",
    "Creo tener habilidad para",
    "Sería muy feliz si",
    "Si la gente trabaja (o trabajara) bajo mis órdenes",
    "Busco",
    "En la escuela, mis maestros",
    "La mayoría de mis amigos no saben que yo temo",
    "No me gusta la gente que",
    "Antes, cuando era más joven",
    "Pienso que la mayoría de las mujeres",
    "Siento que la vida matrimonial es",
    "Mi familia me trata como",
    "Las personas con quienes trabajo son",
    "Mi madre y yo",
    "Mi peor equivocación fue",
    "Deseo que mi padre",
    "Mi mayor debilidad es",
    "Mi ambición secreta en la vida",
    "La gente que trabaja (o trabajara) para mí",
    "Algún día yo",
    "Cuando veo venir a mi jefe",
    "Me gustaría perder el miedo a",
    "La gente que más me gusta",
    "Si otra vez empezara a vivir",
    "Creo que la mayoría de las mujeres",
    "Si tuviera relaciones sexuales",
    "La mayoría de las familias que conozco",
    "Me gustaría trabajar con gente que",
    "Pienso que la mayoría de las madres",
    "De niño me sentía culpable de",
    "Creo que mi padre es",
    "Cuando las circunstancias me son adversas",
    "Cuando doy órdenes, yo",
    "Lo que más deseo en la vida, es",
    "Cuando tenga más edad",
    "Las personas que considero mis superiores",
    "A veces mis temores me impulsan a",
    "Cuando estoy ausente, mis amigos",
    "Mi experiencia infantil más vívida",
    "Lo que menos me gusta de las mujeres",
    "Mi vida sexual",
    "Cuando era niño, mi familia",
    "La gente que trabaja conmigo, generalmente",
    "Me gusta mi madre, pero",
    "Lo peor que he hecho en mi vida es"
  ];

  // Añade cada frase incompleta como una pregunta de párrafo
  for (var i = 0; i < frasesIncompletas.length; i++) {
    form.addParagraphTextItem()
        .setTitle(frasesIncompletas[i] + "...") // Añade puntos suspensivos para indicar que debe completarse
        .setRequired(true);
  }

  // Muestra un mensaje con la URL del formulario
  Logger.log('Formulario creado: ' + form.getEditUrl());
  Logger.log('URL para compartir: ' + form.getPublishedUrl());
}
