
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_x1b95_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_x1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.825827576513755e+00, -5.825835139460535e+00, -5.825886820347782e+00, -5.825773896086390e+00, -5.825830499670503e+00, -5.825830499670503e+00, -1.015439873174114e+00, -1.015429322620741e+00, -1.015190343736551e+00, -1.015586484392139e+00, -1.015354097616446e+00, -1.015354097616446e+00, -2.296472376295309e-01, -2.294448865035113e-01, -2.249540135810726e-01, -2.257971787984398e-01, -2.265668779079854e-01, -2.265668779079854e-01, -6.936773178729372e-02, -7.001008961546870e-02, -2.785069295926942e-01, -5.780322312420581e-02, -6.176884835032877e-02, -6.176884835032875e-02, -1.316281396665990e-02, -1.316417626287885e-02, -2.499410523306739e-02, -1.207410246367659e-02, -1.225281294515867e-02, -1.225281294515867e-02, -1.439410617432553e+00, -1.439255745116377e+00, -1.439405162158623e+00, -1.439268447168126e+00, -1.439331173247154e+00, -1.439331173247154e+00, -6.165516836616728e-01, -6.194203619455831e-01, -6.161519462147734e-01, -6.186813234815655e-01, -6.183381217787711e-01, -6.183381217787711e-01, -1.920334381666245e-01, -2.048332746085818e-01, -1.821816717045640e-01, -1.907529220422362e-01, -1.942099204847468e-01, -1.942099204847468e-01, -4.605393066327239e-02, -7.263239942629122e-02, -4.355880864885323e-02, -5.630099802074308e-01, -4.967311061118151e-02, -4.967311061118151e-02, -1.090819713855012e-02, -1.163140316393173e-02, -7.607754350172654e-03, -3.280688048908736e-02, -9.275000085054603e-03, -9.275000085054608e-03, -1.842190257945553e-01, -1.843977431337226e-01, -1.843441627108928e-01, -1.842924777982607e-01, -1.843194728320422e-01, -1.843194728320422e-01, -1.805759591067996e-01, -1.653178908364023e-01, -1.696571004930426e-01, -1.739061852851886e-01, -1.717452111615432e-01, -1.717452111615432e-01, -2.153858742864064e-01, -8.693753886730186e-02, -9.961612879601166e-02, -1.235075676881650e-01, -1.103169840487955e-01, -1.103169840487955e-01, -1.547658480085231e-01, -2.472187006855633e-02, -2.925053700114709e-02, -1.184639358017298e-01, -3.783704071421592e-02, -3.783704071421592e-02, -1.480679400577701e-02, -6.753103514452245e-03, -8.814102628699715e-03, -3.634824976939227e-02, -8.717938059662214e-03, -8.717938059662209e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_x1b95_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_x1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.998974945080872e+00, -6.998961731075182e+00, -6.999004242805953e+00, -6.998983376157501e+00, -6.999093703953844e+00, -6.999110698265932e+00, -6.998758159546428e+00, -6.998688428769312e+00, -6.998989518710083e+00, -6.998855524793406e+00, -6.998989518710083e+00, -6.998855524793406e+00, -1.212235536585624e+00, -1.212297218487740e+00, -1.212235357552411e+00, -1.212304053076861e+00, -1.212322913754416e+00, -1.212354160829737e+00, -1.212001172342964e+00, -1.212081790048393e+00, -1.212105016146577e+00, -1.212118686523347e+00, -1.212105016146577e+00, -1.212118686523347e+00, -2.881883911333194e-01, -2.888064395596343e-01, -2.875615201811866e-01, -2.884209822518154e-01, -2.755289957341423e-01, -2.736967402357111e-01, -2.745022737587030e-01, -2.745745815284579e-01, -2.831943545324159e-01, -2.684215315785273e-01, -2.831943545324159e-01, -2.684215315785273e-01, -8.015709354459259e-02, -7.725131206537873e-02, -8.191531455731170e-02, -7.859833491303767e-02, -3.685613955444087e-01, -3.671624878933027e-01, -5.958162900029508e-02, -5.869281943669062e-02, -5.840871343045891e-02, -8.167745451507649e-02, -5.840871343045888e-02, -8.167745451507649e-02, -3.626549074496726e-03, -3.749071216081246e-03, -3.698065780053795e-03, -3.839935535128116e-03, -1.709963144811862e-02, -1.802122003569893e-02, -2.851708531543280e-03, -2.840899475409663e-03, -3.410436852302540e-03, -2.305113619675782e-03, -3.410436852302534e-03, -2.305113619675779e-03, -1.777365160223289e+00, -1.777005987309088e+00, -1.777996441672320e+00, -1.777619127385464e+00, -1.777397916624616e+00, -1.777027201822926e+00, -1.777942086947871e+00, -1.777581749880643e+00, -1.777688978958438e+00, -1.777314557192158e+00, -1.777688978958438e+00, -1.777314557192158e+00, -6.770338323801913e-01, -6.769043318028769e-01, -6.816777500390346e-01, -6.814829984473869e-01, -6.715686344693258e-01, -6.722904256944628e-01, -6.756045070817709e-01, -6.763534435144951e-01, -6.830837619386795e-01, -6.808675087175109e-01, -6.830837619386795e-01, -6.808675087175109e-01, -2.410661177660797e-01, -2.406674430180446e-01, -2.566537053894336e-01, -2.563904339735955e-01, -2.311018864589648e-01, -2.289854157098142e-01, -2.358872820022297e-01, -2.361252532684725e-01, -2.443189718867727e-01, -2.438528183787320e-01, -2.443189718867726e-01, -2.438528183787320e-01, -4.267997691635128e-02, -4.233884908056341e-02, -7.581817437503521e-02, -7.561559845385421e-02, -4.045691214679803e-02, -3.963868833726991e-02, -7.020078429157823e-01, -7.017922626324655e-01, -5.029358987835819e-02, -4.629442542807050e-02, -5.029358987835819e-02, -4.629442542807050e-02, -2.395286923290055e-03, -2.485763148134425e-03, -2.730836190166825e-03, -2.779557424957129e-03, -1.891012099955910e-03, -1.916535856137081e-03, -2.958519536910170e-02, -2.969644002233753e-02, -2.098242693077123e-03, -2.441961536496057e-03, -2.098242693077128e-03, -2.441961536496057e-03, -2.362159750613675e-01, -2.364008965224667e-01, -2.359863464220841e-01, -2.361371374351700e-01, -2.361676105220895e-01, -2.363248472396461e-01, -2.362356250193970e-01, -2.364085198482331e-01, -2.362111363582444e-01, -2.363759707734825e-01, -2.362111363582444e-01, -2.363759707734825e-01, -2.296416786735826e-01, -2.298383676581189e-01, -2.005504293297286e-01, -2.005146662179775e-01, -2.105787451846118e-01, -2.105866017834980e-01, -2.199484908994632e-01, -2.200279254343162e-01, -2.153817556358430e-01, -2.154065607680248e-01, -2.153817556358430e-01, -2.154065607680248e-01, -2.689340888636220e-01, -2.684736786560491e-01, -9.503846780704953e-02, -9.471182402414430e-02, -1.148839089605613e-01, -1.141885471538456e-01, -1.542937532390356e-01, -1.542640391500156e-01, -1.325634724096568e-01, -1.325124461512056e-01, -1.325634724096568e-01, -1.325124461512056e-01, -1.901574796726042e-01, -1.896752452120813e-01, -1.644415586421846e-02, -1.655805608999300e-02, -2.386215523806405e-02, -2.438452234191201e-02, -1.521489525178842e-01, -1.492588837664965e-01, -3.588347599875802e-02, -3.394333407965045e-02, -3.588347599875802e-02, -3.394333407965044e-02, -4.558631115623574e-03, -4.644625165036519e-03, -1.334026422980575e-03, -1.217837242729460e-03, -1.881541001469127e-03, -1.912198443317186e-03, -3.342922480843520e-02, -3.331016214533788e-02, -1.856584786721896e-03, -2.328852294051294e-03, -1.856584786721899e-03, -2.328852294051290e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_x1b95_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_x1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.661651655136161e-09, 0.000000000000000e+00, -3.661709748263395e-09, -3.661629062822361e-09, 0.000000000000000e+00, -3.661692933159251e-09, -3.661519511968148e-09, 0.000000000000000e+00, -3.661548504444614e-09, -3.661780974489191e-09, 0.000000000000000e+00, -3.661875634180713e-09, -3.661643801053159e-09, 0.000000000000000e+00, -3.661715571600070e-09, -3.661643801053159e-09, 0.000000000000000e+00, -3.661715571600070e-09, -2.861982924378688e-06, 0.000000000000000e+00, -2.859479175294303e-06, -2.862221152581065e-06, 0.000000000000000e+00, -2.859398636399738e-06, -2.864051132365723e-06, 0.000000000000000e+00, -2.863770855866686e-06, -2.874189319432179e-06, 0.000000000000000e+00, -2.871761010037816e-06, -2.862827091720338e-06, 0.000000000000000e+00, -2.873843588870437e-06, -2.862827091720338e-06, 0.000000000000000e+00, -2.873843588870437e-06, 1.158062181641945e-03, 0.000000000000000e+00, 1.323682722364579e-03, 1.092444907315667e-03, 0.000000000000000e+00, 1.300554436956595e-03, 2.765127939355579e-04, 0.000000000000000e+00, 4.301864103114627e-05, 2.238181007067300e-06, 0.000000000000000e+00, 3.945160204264905e-05, 1.465350523898422e-03, 0.000000000000000e+00, -7.061342941558839e-04, 1.465350523898422e-03, 0.000000000000000e+00, -7.061342941558839e-04, -1.245541880529634e-01, 0.000000000000000e+00, -1.514298538434611e-01, -1.067575274899006e-01, 0.000000000000000e+00, -1.381989952888980e-01, 2.346733568167524e-03, 0.000000000000000e+00, 2.479740321228544e-03, -4.143876356025827e-01, 0.000000000000000e+00, -4.277512035962466e-01, -2.812682853087389e-01, 0.000000000000000e+00, 6.463823102519259e-01, -2.812682853087387e-01, 0.000000000000000e+00, 6.463823102519279e-01, -4.492435183850389e+03, 0.000000000000000e+00, -3.762176700456231e+03, -3.984805826491042e+03, 0.000000000000000e+00, -3.264998770606959e+03, -2.046578896108555e+01, 0.000000000000000e+00, -1.736389969642089e+01, -1.897643184224578e+04, 0.000000000000000e+00, -1.976531911034401e+04, -6.238391651669291e+03, 0.000000000000000e+00, -4.102813556213558e+04, -6.238391651669296e+03, 0.000000000000000e+00, -4.102813556213555e+04, -9.012713808191466e-07, 0.000000000000000e+00, -9.020338885474546e-07, -9.012653944042119e-07, 0.000000000000000e+00, -9.020232902822872e-07, -9.012862557782266e-07, 0.000000000000000e+00, -9.020417637930176e-07, -9.012767535676546e-07, 0.000000000000000e+00, -9.020354525983920e-07, -9.012589528086276e-07, 0.000000000000000e+00, -9.020261140104844e-07, -9.012589528086276e-07, 0.000000000000000e+00, -9.020261140104844e-07, -2.838634435657222e-05, 0.000000000000000e+00, -2.842043876669975e-05, -2.785294358656435e-05, 0.000000000000000e+00, -2.789595672892724e-05, -2.886942078781488e-05, 0.000000000000000e+00, -2.884294827738030e-05, -2.839776525150249e-05, 0.000000000000000e+00, -2.836624478698097e-05, -2.777839263739764e-05, 0.000000000000000e+00, -2.799317942981905e-05, -2.777839263739764e-05, 0.000000000000000e+00, -2.799317942981905e-05, 1.762890201905823e-03, 0.000000000000000e+00, 1.728939574495058e-03, 5.961657380787672e-03, 0.000000000000000e+00, 5.850203755740846e-03, 5.299279720646066e-03, 0.000000000000000e+00, 3.394970088765348e-03, 1.919989164379681e-02, 0.000000000000000e+00, 1.329872545749686e-02, 1.275620082897478e-03, 0.000000000000000e+00, 2.781037688456880e-03, 1.275620082897473e-03, 0.000000000000000e+00, 2.781037688456876e-03, -1.212882548500654e+00, 0.000000000000000e+00, -1.221612437103907e+00, -1.626320858061401e-01, 0.000000000000000e+00, -1.627244563069485e-01, -1.536910817496794e+00, 0.000000000000000e+00, -1.481769119899947e+00, 2.044458643869975e-05, 0.000000000000000e+00, 2.054148985560887e-05, -7.879935908761573e-01, 0.000000000000000e+00, -9.253461180654285e-01, -7.879935908761573e-01, 0.000000000000000e+00, -9.253461180654287e-01, -4.778219641002914e+04, 0.000000000000000e+00, -4.107329712310053e+04, -2.236083391309653e+04, 0.000000000000000e+00, -2.094101661005439e+04, -8.404427510162734e+04, 0.000000000000000e+00, -7.341131205214167e+04, -4.451791830526997e+00, 0.000000000000000e+00, -4.346190527934066e+00, -6.503699949632462e+04, 0.000000000000000e+00, -2.595962630119599e+04, -6.503699949632455e+04, 0.000000000000000e+00, -2.595962630119598e+04, 5.477424898758240e-03, 0.000000000000000e+00, 5.318524677533890e-03, 3.888045522415560e-03, 0.000000000000000e+00, 3.785664854748202e-03, 4.432761998450189e-03, 0.000000000000000e+00, 4.316592309662002e-03, 4.909897130330954e-03, 0.000000000000000e+00, 4.771667908656789e-03, 4.670979108980634e-03, 0.000000000000000e+00, 4.543745297422025e-03, 4.670979108980634e-03, 0.000000000000000e+00, 4.543745297422025e-03, 8.489371314037979e-03, 0.000000000000000e+00, 8.216055949601382e-03, -1.999009271103391e-03, 0.000000000000000e+00, -2.024697453670775e-03, -5.598547282097133e-05, 0.000000000000000e+00, -8.464776349203194e-05, 2.575210244968268e-03, 0.000000000000000e+00, 2.499909202875796e-03, 1.218051069508303e-03, 0.000000000000000e+00, 1.129086090984877e-03, 1.218051069508303e-03, 0.000000000000000e+00, 1.129086090984881e-03, 5.322930965873153e-03, 0.000000000000000e+00, 5.103889269319224e-03, -6.830991636946650e-02, 0.000000000000000e+00, -6.879110149070955e-02, -2.865326720234522e-02, 0.000000000000000e+00, -2.952708481673394e-02, 7.934685581231823e-03, 0.000000000000000e+00, 7.740330919912712e-03, -8.946894226908385e-03, 0.000000000000000e+00, -8.787928014421249e-03, -8.946894226908385e-03, 0.000000000000000e+00, -8.787928014421256e-03, -2.515619194994306e-04, 0.000000000000000e+00, -5.296767944262606e-04, -2.156707483359733e+01, 0.000000000000000e+00, -2.112538684209944e+01, -8.080438906332652e+00, 0.000000000000000e+00, -7.448862003339674e+00, 3.047645469454529e-02, 0.000000000000000e+00, 2.029678820340136e-02, -2.503926887243264e+00, 0.000000000000000e+00, -2.715700446175924e+00, -2.503926887243265e+00, 0.000000000000000e+00, -2.715700446175926e+00, -1.502319618842738e+03, 0.000000000000000e+00, -1.364070127082485e+03, -1.412944498258606e+06, 0.000000000000000e+00, -1.644114626431146e+06, -1.585253608448432e+05, 0.000000000000000e+00, -1.350161329022057e+05, -3.071493352497328e+00, 0.000000000000000e+00, -3.013981301265816e+00, -9.309276321386090e+04, 0.000000000000000e+00, -3.370897500079434e+04, -9.309276321386112e+04, 0.000000000000000e+00, -3.370897500079445e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_x1b95_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_x1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_x1b95_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_x1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.453789028236753e-05, -1.453795539533437e-05, -1.453819599291548e-05, -1.453817725604425e-05, -1.453863452856280e-05, -1.453895670951458e-05, -1.453519611389047e-05, -1.453456982004614e-05, -1.453809985687147e-05, -1.453592645007412e-05, -1.453809985687147e-05, -1.453592645007412e-05, -2.773989440474821e-04, -2.777448485755736e-04, -2.774399024960354e-04, -2.778084148197651e-04, -2.786639438061573e-04, -2.789831013985793e-04, -2.764684431128833e-04, -2.768336125599870e-04, -2.773459159014582e-04, -2.778996350686698e-04, -2.773459159014582e-04, -2.778996350686698e-04, -2.491933376166078e-03, -2.642184406024723e-03, -2.443485038052027e-03, -2.625098956656117e-03, -1.892260784642392e-03, -1.718606872764310e-03, -1.784285750730145e-03, -1.842295697882058e-03, -2.877173658706024e-03, -1.074779912048118e-03, -2.877173658706024e-03, -1.074779912048118e-03, -2.393230896972181e-03, -2.713708004130124e-03, -2.615544232205190e-03, -3.002092192391819e-03, -2.055021085824489e-03, -2.505285064109242e-03, -9.520634906531265e-04, -9.656735293285018e-04, -2.354546618229709e-03, -4.301318033992195e-04, -2.354546618229706e-03, -4.301318033992202e-04, -2.997087148185276e-07, -3.518866688465333e-07, -3.601478287499169e-07, -4.346246878873608e-07, -2.296147427444176e-05, -2.686840224581216e-05, -6.513783696351832e-08, -6.041348501259918e-08, -2.169816319607273e-07, -1.113827375369721e-07, -2.169816319607280e-07, -1.113827375369724e-07, -2.269464053455427e-04, -2.269942041936558e-04, -2.280519143712854e-04, -2.280628082763925e-04, -2.270020733267227e-04, -2.270268297199039e-04, -2.279506377716149e-04, -2.279975920571464e-04, -2.275163759208161e-04, -2.275310559213020e-04, -2.275163759208161e-04, -2.275310559213020e-04, -2.646821730286466e-04, -2.646819876534644e-04, -2.699096506917164e-04, -2.698297537548436e-04, -2.456280887323677e-04, -2.511417099558468e-04, -2.500045673272401e-04, -2.554573505090169e-04, -2.824632236149086e-04, -2.694761276964254e-04, -2.824632236149086e-04, -2.694761276964254e-04, -6.347379240705518e-03, -6.390511160701925e-03, -1.184698105674906e-02, -1.195811997073223e-02, -6.321235538715860e-03, -6.490396976230268e-03, -1.327119950005738e-02, -1.243099175812918e-02, -6.676694104181627e-03, -6.989931597318513e-03, -6.676694104181627e-03, -6.989931597318513e-03, -2.781922350892201e-04, -2.928988132952843e-04, -1.076133359808841e-03, -1.086732839708232e-03, -2.304560833431565e-04, -2.641079765903469e-04, -1.927400997080872e-03, -1.929892281417028e-03, -4.987911163701257e-04, -7.898358953142875e-04, -4.987911163701257e-04, -7.898358953142875e-04, -4.030688190349819e-08, -3.894187119486963e-08, -7.056855839852068e-08, -6.799578383656435e-08, -1.755342139979134e-07, -2.301047862615557e-07, -1.134151689771698e-04, -1.109104482960024e-04, -9.269830704810115e-08, -2.249914142493880e-07, -9.269830704810089e-08, -2.249914142493877e-07, -1.717652751912049e-02, -1.713619248568862e-02, -1.386428745944997e-02, -1.386672723675737e-02, -1.491063312506914e-02, -1.490782769044810e-02, -1.587685030660696e-02, -1.584744786659725e-02, -1.538304991477683e-02, -1.536705133539403e-02, -1.538304991477683e-02, -1.536705133539403e-02, -1.929224223085034e-02, -1.920034085401544e-02, -4.370113379078337e-03, -4.409813643531602e-03, -6.427959367186009e-03, -6.497529364432018e-03, -9.824749734082082e-03, -9.829088996211842e-03, -7.943079478903152e-03, -7.949467297785930e-03, -7.943079478903152e-03, -7.949467297785928e-03, -1.059149578002199e-02, -1.072681605893395e-02, -1.540683307611430e-03, -1.564565636171346e-03, -2.429583102321081e-03, -2.531861492405219e-03, -6.287727115763946e-03, -6.323005669183542e-03, -4.006584648178532e-03, -4.059123384598535e-03, -4.006584648178535e-03, -4.059123384598537e-03, -4.805309699537307e-03, -4.900319655551454e-03, -1.988346384035491e-05, -2.028650905216560e-05, -4.476415823168420e-05, -5.025037174718308e-05, -9.841532611841538e-03, -1.007052461472450e-02, -2.184009360754973e-04, -3.122002109861319e-04, -2.184009360754975e-04, -3.122002109861319e-04, -6.466774595219028e-07, -7.290699924745556e-07, -5.755693045808994e-09, -1.026365461509461e-08, -2.775916785357306e-08, -3.528415037704372e-08, -2.162876966262462e-04, -2.221903081659497e-04, -1.636920504245606e-07, -1.823002569306333e-07, -1.636920504245608e-07, -1.823002569306335e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05