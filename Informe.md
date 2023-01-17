#  <center>Proyecto de IA y Simulación  </center>
# <center>I.F.S.L</center> 

## Equipo:

- Tony Cadahía Poveda C312
- Alejandra Monzón Peña C312
- Leonardo Ulloa Ferrer C312
  
## Descripción de proyecto

IFSL (Intellient Formation Simulation Language) es una simulación de batallas de un ejército, con características similares a muchos juegos de guerras medievales existentes. El principal objetivo es estudiar el comportamiento de los agentes de la simulación con diferentes estrategias de IA, analizar acciones de combates, exploración y formación de las unidades. Las unidades presentan diferentes características respecto a su radio de visión, poder de ataque, costo de movilidad entre otras, de modo que pueden enfocarse como un conjunto de robots con diferentes composiciones.

## Elementos de la simulación

### Características del Ambiente

Nuestro medio ambiente consiste en un grid bidimensional en el que pueden o no haber obstáculos. Los agentes tienen una posición en el mapa y pueden interactuar con el ambiente decidiendo a qué posición moverse y a donde atacar.

La simulación transcurre por turnos en la que todos los agentes actúan "simultáneamente". Para lograr esto los turnos se dividen en 3 momentos, uno en el que los agentes deciden que acción van a tomar (a que dirección moverse o a qué posición atacar), en un segundo momento se realizan los movimientos y en un 3ro se realizan los ataques.

Para simular el hecho de que distintos tipos de agentes tengan distintas velocidades añadimos el concepto de costo de movimiento en el cual un agente debe esperar el costo de movimiento correspondiente a su tipo para poder avanzar una posición.

El intento de lograr la simultaneidad en las acciones de los agentes trajo consigo varias situaciones en la que los resultados de las acciones de los agentes dependen de una condición de carrera, a los cuales le definimos que el resultado lo decide una variable aleatoria que distribuye binomial en dependencia del costo de movimiento de los involucrados, favoreciendo a los más rápidos. Este es el caso en el que dos agentes deciden moverse a una misma casilla y les toca moverse en el mismo turno.

Otro problema de la simultaneidad es decidir cuándo un movimiento es válido en los casos que el agente decide ir a posiciones donde hay otros agentes, pues este se puede mover a esa casilla si no hay nadie en el momento que se va a mover, pero a su vez el hecho de que él se mueva puede depender de otro. Para resolver esto se construyó el grafo de dependencias priorizando los movimientos que son ciclos y resolviendo como se explicó anteriormente los casos en que más de uno se quiera mover a una misma casilla.

La implementación del medio ambiente se basa en 4 objetos, un mapa que contiene el estado, un StateManager que se encarga de decidir qué acciones son válidas y realizar las modificaciones al estado correspondientes a las acciones, un agente que se encarga de tomar decisiones y recibir percepciones.

El escenario de la simulación tiene las siguientes características.

- Inaccesible: Se obtiene información solo de la parte visible del mapa por el agente, se conoce sobre la geografía del entorno, pero no sobre la locación de los oponentes hasta que entren en el rango de visión.
  
- No determinista: Las acciones de ataque no tienen siempre los mismos resultados, van a depender de si las unidades atacadas se encuentran en movimiento o no y de las velocidades del atacante y el atacado, de igual forma si dos unidades se desean mover a la misma posición en el mapa no siempre se dará prioridad de movimiento a la misma unidad.
  
- No episódico: Las acciones en los combates se planifican en vista a futuro, de igual forma hay recorridos en el mapa en los que se contempla toda una trayectoria.
  
- Estático: Los agentes son los únicos que modifican el medio mediante sus acciones.

- Discreto: Las únicas posibles acciones son moverse, atacar o esperar, por tanto, el conjunto de estados posibles es finito y está dado por la combinación de estas acciones.

### Arquitectura del Agente

La arquitectura de Agente utilizada es mixta, se emplean 3 capas verticales de las cuales la capa intermedia está conformada como dos capas horizontales.

Además, se tiene un subsistema de percepción que se comunica con la capa superior.

Las diferentes capas se comunican con un sistema de eventos, de modo que se pueden ver como Agentes que interactúan entre si informándose sobre qué acciones realizar

#### Capa Superior

La capa superior presenta el comportamiento de la arquitectura de Brook, donde en dependencia de la percepción y el estado actual se decide una acción. Al principio se asignan los exploradores a partir de las formaciones existentes y se procede a explorar el área, para ello se tiene en cuenta la cantidad de unidades y el costo de movimiento (seleccionando los más rápidos para esta tarea).
  
Las acciones a realizar son “ordenes” que se envían a la capa media. Estas se dividen en atacar, moverse, formar y explorar; se tiene un diccionario de misiones por cada una de las unidades y se van limpiando a medida que se completan. Además, existe un sistema de eventos que debe de notificar el agente intermedio para saber cuándo se concluye una misión y poder asignarle otra o pasar a la siguiente.

Existe un orden de prioridad entre las acciones, ejemplo de esto es que al momento de ver un enemigo se limpia la lista de misiones de esa unidad, se procede a avisar a las demás de que existe ese enemigo y entra automáticamente al combate. Otra prioridad es moverse en formación al objetivo antes de cambiar a modo de combate, aunque si se encuentran otro enemigo hacen el mismo procedimiento anterior.

#### Capa Media

La capa media está conformada por dos capas horizontales, una especializada en tareas de movimiento de unidades en el mapa y la otra en tomar decisiones en combates. La primera tiene todos los algoritmos anteriormente descritos para explorar, moverse en formación y formar.

Estos agentes planifican las acciones a realizar teniendo en cuenta el objetivo que persiguen y cómo esto podría afectar el medio. Cada cambio existente (como perder una unidad o que no exista más ninguna) es notificada a través de eventos.

Estos agentes toman acciones en dos sentidos, le indican a la capa baja órdenes a seguir y le informan a la capa alta sobre las decisiones tomadas. Estas órdenes difieren entre seguir un camino si es la capa de movimiento o luchar si es la capa de combate, debe recibir notificaciones de la capa baja en caso de que algún agente no pueda moverse o que alguna unidad muera en batalla.

#### Capa Baja

La capa baja es el Agente que realmente modifica el medio, recibe como percepción la planificación de la capa media e intenta ejecutar las acciones correspondientes con el fin de lograr el objetivo.

Esta capa recibe cierta información directamente del medio, como si la acción que ejecutó fue efectiva o si la unidad que maneja permanece con vida, la información relevante se le comunica a la capa media para que re-planifique en caso de ser necesario.

## Principales problemas y propuestas de solución

Entre los principales problemas encontrados durante el desarrollo del proyecto se encuentran:

- Explorar el territorio en busca de enemigos
- Asignarle a las unidades posiciones en las formaciones
- Mover las unidades a sus posiciones en la formación
- Mover las formaciones hacia los enemigos y otros objetivos específicos
- Evitar obstáculos
- Luchar contra los enemigos
  
Para cada una de estas problemáticas se utilizaron algoritmos de publicaciones destacadas sobre cada tema en específico. A continuacion presentamos un breve resumen de cada una de las ideas utilizadas y los resultados obtenidos en la simulación.

### Explorar el territorio en busca de enemigos

Nuestro problema de explorar en el mapa es equivalente a un problema bien trabajado, conocido como Coverage Path Planning en el que se busca un camino que pase por todos los puntos del mapa evitando los obstáculos.

Hay varios acercamientos para intentar resolver este problema, algunos se basan en caminos aleatorios, como es el caso del algoritmo que utilizan los Roomba. También hay otros enfoques como el Spiral STC donde se bordea toda el área y se va cubriendo de afuera hacia adentro; pero la mayoría de soluciones se basan en descomponer el área a explorar en celdas. Una de estas descomposiciones es la de "Boustrophedon"[1] (camino de la oz en griego) el cual descompone el mapa en áreas que pueden ser recorridas con un barrido de arriba hacia abajo.

Una vez obtenida la descomposición surge el problema de escoger el orden en que se van a recorrer las celdas, el cual es muy parecido a TSP, lo que también se pueden escoger los puntos de inicio lo que también se pueden escoger los puntos de inicio del recorrido de cada célula

Como el espacio de búsqueda es muy grande, la mayoría de algoritmos que resuelven este problema sufren de una gran complejidad computacional, por lo que optamos por una alternativa basada en algoritmos genéticos [5]. Esto nos permite obtener soluciones suficientemente buenas en menos tiempo como se muestra en el artículo citado.

En este algoritmo genético los individuos son permutaciones de las células. La población inicial se obtiene usando Knuth Shuffle un algoritmo que entrega permutaciones aleatorias de los elementos con una distribución uniforme. Como función de fitness se calcula el costo aproximado de explorar las celdas con el path que representa el individuo para la mejor selección de configuraciones para las celdas, que se calcula con una DP en tiempo lineal de la cantidad de celdas, el costo de explorar la celda ya viene calculado de la descomposición y el costo de moverse de una célula a otra se aproxima con la norma infinito entre el punto donde termina el recorrido de esa célula y empieza el de la otra. Para el crossover usamos Roulette Wheel para la selección y como operación para crear el nuevo individuo se utiliza una  modificación de una operación ya probada para TSP que consiste en seleccionar una celda de inicio random y  a partir de los cuatro adyacentes que puede tener esta en los dos individuos  y las 4 posibles configuraciones que puede tener cada una de sus células escoger como siguiente  la que minimice la función de costo que consiste en el costo del camino intercelular sumado a la diferencia del costo de explorarlo con esa configuración con el costo mínimo de explorarlos teniendo en cuenta las 4 configuraciones, repitiendo esta operación se construye el nuevo individuo. Como operación de mutación utilizamos el random swap y un algoritmo de búsqueda local para TSP llamado 2-opt que consiste en encontrar dos vértices en el camino que swapeados mejoran el costo

### Asignación de posiciones en las formaciones

Otro problema es decidir dada una formación y las unidades dispersas en el mapa que posición de la misma deben ocupar. En un principio se puede pensar en la idea de asignar aleatoriamente las posiciones o asignar posiciones fijas por numeración, pero esto en ocasiones produce muchas intersecciones entre los recorridos de las unidades y se tienen que generar desvíos en las trayectorias lo que enlentece el proceso de formado, por este motivo decidimos representar el problema de asignar posiciones como un problema de satisfacción de restricciones. Este enfoque basa en las ideas utilizadas para resolver el problema de la N reinas.

#### CSP

Representar el problema como en CSP que dados los puntos de ubicación actual de las unidades y los puntos de las posiciones en que debe haber alguien en la formación devuelva una asignación en la que para cada punto de destino hay un único punto al cual debe llegar y recíprocamente a cada destino debe llegar solo una unidad. Además, se pone como restricción que las rectas que se forman entre los puntos unidad posición que sean asignados no se intercepten con los de ninguna otra asignación.

Poner que explícitamente no haya intercepciones entre las rectas es una condición un poco fuerte, cuando realmente también hay que considerar que tal vez las rectas se interceptan, pero con distancias entre el lugar donde están posicionadas las unidades respecto al punto de intersección lo suficientemente grandes como para que a una unidad le dé tiempo a pasar sin que choque realmente con la otra. Por tanto, la restricción real utilizada es que las restas no se intercepten en un punto tal que la diferencia entre las distancias de las ubicaciones de las unidades a ese punto sea menor que una tolerancia establecida. 

No obstante, quedan casos en lo que esta asignación no tiene solución posible.


#### Hill Climbing

  Para los casos en los que el CSP no funciona utilizamos Hill Climbing, la primera asignación se crea random con la función shuffle de python. Para encontrar un nuevo candidato se compara con soluciones iguales exceptuando 2 de posiciones asignadas o, lo que es lo mismo, todas las formas de escoger 2 posiciones finales e intercambiar su posición inicial. Para el caso de parada se decidió hacer mínimo 30 iteraciones o detenerse si no se encontró un nuevo candidato y para la función de fitness se trata de minimizar la cantidad de intercepciones.

### Mover a las unidades a sus posiciones en la formación (WHCA*)

Una vez que cada agente sabe qué posición de la formación le corresponde tenemos que resolver el problema de encontrar una ruta en la cual no se encuentren en la misma posición y al mismo tiempo. Para ello se introduce la idea de tener un espacio vectorial de 3 dimensiones (2 espaciales y una temporal) en el cuál se representan los caminos que pueden tener los agentes, de esta forma podemos calcular la ruta de un agente y luego tenerla en cuenta para el cálculo de la próxima ruta [10].

El problema de esta idea es el enorme espacio de búsqueda, algoritmos como A\* son ineficientes en simulaciones donde se quieren tener gran cantidad de agentes. Para simplificar el problema utilizamos una abstracción del mismo utilizando la idea de espacios jerárquicos, que no es más que realizar la búsqueda en un espacio más simplificado (en nuestro caso eliminamos la variable temporal) para poder resolver con más facilidad el problema actual [4].

Esta idea la usamos específicamente como cálculo de la heurística de nuestro problema utilizando el algoritmo RRA\*, este consiste en realizar un recorrido de A\* pero en sentido inverso, calculando así el costo del camino real desde el punto de llegada hasta el punto desde donde se quiere saber qué costo tiene la heurística.

Como aplicamos este recorrido sobre la abstracción del problema original, tendríamos como heurística el costo camino real desde el punto hasta la posición asignada (heurística que es admisible porque al menos tienes que recorrer ese camino para llegar al objetivo, y posiblemente se demore más en llegar porque el agente se tiene que desviar si hay alguien en dicho camino), para la heurística de este RRA\* se utilizó la norma infinito, ya que en nuestra simulación se tiene en cuenta que se puede mover por las diagonales.

A pesar de todos estos arreglos al algoritmo de A\*, seguimos teniendo un costo computacional bastante elevado, para reducir esto introducimos la idea de ventanas de tiempo, esto nos permite calcular solo un tramo del camino y ejecutarlo a medida que va realizándose la simulación, en los casos en que el objetivo no cambie se puede reutilizar los cálculos realizados por el RRA\*.

### Movimiento de las formaciones hacia objetivos específicos

Ya teniendo a los agentes formados podemos pasar al problema de caminar hacia el enemigo evitando los obstáculos, para ello se tienen varias ideas como utilizar Potencial Field Method [3] o discretizar el espacio utilizando RoadMaps, nosotros escogimos este último que se divide en calcular el grafo de visión reducida o utilizar el grafo de Voronoi [4]. Como nuestro objetivo es mantenernos alejados de los obstáculos para evitar romper la formación por posibles colisiones, el grafo de visión no nos sirve ya que los puntos que escoge son los más cercanos a los obstáculos con el objetivo de bordearlos.

Para hacer un grafo de Voronoi se tiene que cumplir que los puntos centros de cada región equidisten de las aristas que separan a esta región de las demás, de forma tal que cada arista separa 2 regiones y cada punto separe a 3 o más regiones. Con esto en mente utilizamos un BFS por todos los obstáculos marcando las celdas contiguas con un número que representaría el "color" de dicha área, y también le asignamos la altura a la que encontramos esa celda en la búsqueda, de esta forma sabemos que distancia tiene hacia el obstáculo más cercano y si estamos en una arista o un vértice (por cuantos colores tenga la celda), para esto tuvimos en cuenta el hecho de detenernos cuando pintáramos una celda que ya se había pintado antes de otro color.

Teniendo ya una representación reducida del mapa en la cual podemos movernos entre obstáculos solo queda saber en qué área del grafo nos encontramos y enlazar el lugar de salida con todos los vértices de Voronoi de esa área para incorporarnos al grafo y empezar el recorrido, lo mismo con el lugar de llegada. Solo queda ejecutar A\* sobre dicho grafo para encontrar el camino, para el costo real utilizamos la longitud de la arista entre los puntos del grafo y para los puntos adicionales se calcula este costo a partir de un BFS sobre el área de Voronoi correspondiente.

### Combate entre ejércitos

Entre las ideas más utilizadas en "juegos" de combate con múltiples jugadores y en equipos se encuentra el aprendizaje reforzado, aunque esta idea es la que más destaque, se sale de las intenciones de probar ideas de IA clásica, otro de los acercamientos más comunes es ver el enfrentamiento con factores estocásticos [12] para así determinar la mejor decisión a tomar.

En el proyecto decidimos darle un enfoque de MiniMax [8, 9] con incorporación de elementos probabilísticos. Como para cada unidad se tiene un conjunto de acciones posibles (Moverse a sus adyacentes desocupados, atacar a los enemigos alcanzables y esperar) aunque parezcan pocos estados, ciertamente sería necesario comprobar todas las posibles combinaciones de estados para las unidades de un mismo grupo ya que juegan en colaboración. Al no ser posible realmente representar todas las posibles acciones dado un estado del juego, la primera estrategia utilizada fue simplificar el problema a combates de 1 contra 1, es decir cada Agente asumía que las unidades se enfrentaban cada una contra una unidad específica rival y hacer MiniMax sobre los posibles estados. Este enfoque al ser simulado no ofreció muy buenos resultados, pues si bien se situaban las unidades enemigas muy cercanas se producía el enfrentamiento pues entraban dentro de los radios de ataque mutuamente, pero si existía mucha distancia entre los bandos opuestos el enfrentamiento no se producía cuando se tenían enemigos estáticos(torres) contra soldados, pues al considerarse solo el enfrentamiento uno a uno el soldado ante la superioridad de la torre veía siempre desventajas en acercarse, mientras que la torre no tiene permitido el movimiento.

Por estas cuestiones al MiniMax le agregamos una percepción de equipo, haciendo uso de la visión cada unidad reconoce cuantos aliados y enemigos tiene cerca, entonces en el MiniMax, aplicar una acción tiene en cuenta que sus aliados también pueden aplicar acciones y se determina la probabilidad de que estas acciones sean disparos a enemigos como:

$$ Pd = \frac{Ner}{Mp + Ner} $$

Donde Ner es el número de enemigos que están en el radio de ataque de la unidad y Mp el total de movimientos posibles de la unidad (casillas adyacentes desocupadas y sin obstáculos contando la opción de mantenerse en el lugar), se generan variables aleatorias con distribución Bin(Pd) para cada unidad aliada y se resta el total de puntos de vida al enemigo correspondiente a estas probabilidades al aplicar el estado en el MiniMax.

El MiniMax se realiza con una altura de corte 3, debido a las características no deterministas de las acciones planificar con una vista a mayor profundidad no producía mejoras significativas en el comportamiento del algoritmo al ejecutarse la simulación.

Con esta forma de implementación de MiniMax se logró una simulación de batalla con actuación interesante por parte de los Agentes, ya que por ejemplo, cuando se encuentran en minoría esperan a la llegada de refuerzos.

Como heurística para el corte en el MiniMax tuvimos en cuenta la diferencia de vida entre el estado inicial del enemigo y en el que se produce el corte, mantener a los enemigos en el campo de visión, mantener a los enemigos en el campo de ataque y si es posible estar fuera del campo de ataque del enemigo.

## Conclusiones

El desarrollo de este proyecto nos ha incursionado en numerosos temas de interés, puesto que las ideas utilizadas son tendencia en el mundo de la robótica, el manejo automático de vahículos, la IA clásica y la simulación en general. Se logró experimentar con los diferentes algoritmos y analizar el comportamiento que estos generan en diferentes circunstancias así como sus ventajas y puntos débiles.

## Bibliografía

- [1] Enric Galceran, Marc Carreras. A Survey on Coverage Path Planning for Robotics. September 24, 2013.
- [2] Frank E. Schneider and Dennis Wildermuth. A potential field based approach to multi robot formation navigation. October 2003.
- [3] Franz Aurenhammer.  Voronoi Diagrams — A Survey of a Fundamental Geometric Data Structure. September 1991.
- [4] R.C. Holte, M.B. Perez, R.M. Zimmer, A.J. MacDonald. Hierarchical A*: Searching Abstraction Hierarchies Efficiently.
- [5] Rutuja Shivgan and Ziqian Dong. Energy-Efficient Drone Coverage Path Planning using Genetic Algorithm.
- [6] O. Burchan Bayazit, Jyh-Ming Lien, Nancy M. Amato. Better Group Behaviors in Complex Environments using Global Roadmaps.
- [7] David Silver. Cooperative Pathfinding. 2005.
- [8] Dilian Gurova, Valentin Gorankob, Edvin Lundbergd. Knowledge-based strategies for multi-agent teams playing against Nature.
- [9] Dhruva Kartik, Ashutosh Nayyar, Urbashi Mitra. Common Information Belief based Dynamic Programs for Stochastic Zero-sum Games with Competing Teams.
- [10] Zahy Bnaya and Ariel Felner. Conflict-Oriented Windowed Hierarchical Cooperative A\*.
- [11] O. Takahashi, R. Schilling. Motion Planning in a Plane Using Generalized Voronoi Diagrams.
- [12] Alexander Shapiro, Anton Kleywegt. Minimax analysis of stochastic problems. 27 Oct 2010.
