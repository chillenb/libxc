
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_gap_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.521154143856828e+01, -2.521158340303242e+01, -2.521191986221343e+01, -2.521129692764196e+01, -2.521160199901700e+01, -2.521160199901700e+01, -3.398718344952502e+00, -3.398966847474905e+00, -3.405878941784437e+00, -3.406426090678175e+00, -3.404740952478366e+00, -3.404740952478366e+00, -4.974701510988728e-01, -4.962470881640935e-01, -4.696708810532633e-01, -4.897817284431114e-01, -4.841831809053224e-01, -4.841831809053224e-01, -1.353128289712902e-01, -1.392508406348790e-01, -4.459958321717922e-01, -7.949402861436146e-02, -9.746382944689264e-02, -9.746382944689264e-02, -3.212374027815696e-03, -3.378309346008412e-03, -1.858742166744575e-02, -1.828115873409177e-03, -2.311941651547736e-03, -2.311941651547735e-03, -6.069519869779131e+00, -6.067886953275827e+00, -6.069497284250494e+00, -6.068053940560879e+00, -6.068666869500026e+00, -6.068666869500026e+00, -2.071120375245034e+00, -2.116794815139379e+00, -2.052802870398881e+00, -2.094562182800400e+00, -2.105136007936527e+00, -2.105136007936527e+00, -6.508484173793241e-01, -6.888124224592144e-01, -5.561197161016508e-01, -5.717517009962769e-01, -6.654902843358503e-01, -6.654902843358503e-01, -5.086458984026198e-02, -1.180097997176009e-01, -4.657249086876366e-02, -1.962482389063672e+00, -6.365779693883571e-02, -6.365779693883571e-02, -1.344975904334945e-03, -1.750427945319071e-03, -1.363377725377557e-03, -3.047256535474402e-02, -1.619653999396204e-03, -1.619653999396204e-03, -6.411114534782600e-01, -6.535137765622013e-01, -6.493973614821372e-01, -6.457803263722451e-01, -6.476098690222207e-01, -6.476098690222207e-01, -6.178620932213656e-01, -5.781889960353868e-01, -6.115478013749154e-01, -6.249620210880832e-01, -6.198867537155686e-01, -6.198867537155686e-01, -7.129856751870366e-01, -1.683682266836116e-01, -2.252352145567838e-01, -3.445207110431809e-01, -2.876255810469217e-01, -2.876255810469218e-01, -4.932661176332824e-01, -1.792046219521655e-02, -2.432619978757269e-02, -3.538348379835109e-01, -4.005463144119416e-02, -4.005463144119418e-02, -4.369537068110291e-03, -4.238281446567536e-04, -1.009957098211635e-03, -3.777175918092428e-02, -1.511953074818919e-03, -1.511953074818917e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_gap_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.858474888049966e+01, -2.858479064876270e+01, -2.858490042185505e+01, -2.858427183759842e+01, -2.858461201126535e+01, -2.858461201126535e+01, -5.411169303574588e+00, -5.411167812345700e+00, -5.411148263158013e+00, -5.411580406749160e+00, -5.411006000593700e+00, -5.411006000593700e+00, -8.868990950210639e-01, -8.839595478136048e-01, -8.161597859815021e-01, -8.523586168883553e-01, -8.433118295489974e-01, -8.433118295489974e-01, -2.197479143099079e-01, -2.273468518508103e-01, -7.645946032536822e-01, -1.106621238705586e-01, -1.451474744736251e-01, -1.451474744736251e-01, -3.870642862372793e-03, -4.067818549375548e-03, -2.243086793998495e-02, -2.181416732272433e-03, -2.768782173485911e-03, -2.768782173485911e-03, -6.910823788258277e+00, -6.913302074398078e+00, -6.910862626914398e+00, -6.913053497177968e+00, -6.912112124159292e+00, -6.912112124159292e+00, -3.139053464161857e+00, -3.162103193802677e+00, -3.132340293450752e+00, -3.155639235544118e+00, -3.154473338974602e+00, -3.154473338974602e+00, -8.624280980429611e-01, -9.171278463762390e-01, -8.388565018196205e-01, -8.632558995102777e-01, -8.647642185426844e-01, -8.647642185426844e-01, -6.490192845519063e-02, -1.779554050315343e-01, -5.885095181920712e-02, -2.911549246481234e+00, -8.612247606526072e-02, -8.612247606526066e-02, -1.591870670619028e-03, -2.074494509727009e-03, -1.625138753030298e-03, -3.737617018957230e-02, -1.921420788462590e-03, -1.921420788462592e-03, -8.294892745096204e-01, -8.058107159948913e-01, -8.135403181229979e-01, -8.204359655210764e-01, -8.169339549253485e-01, -8.169339549253485e-01, -8.149365060900531e-01, -7.369495922066219e-01, -7.182308120002157e-01, -7.275345243400247e-01, -7.182695211850283e-01, -7.182695211850282e-01, -9.773549199466357e-01, -2.714770580410328e-01, -3.738631483030739e-01, -5.447057108824663e-01, -4.691505397856677e-01, -4.691505397856679e-01, -7.061171136560755e-01, -2.171015002748405e-02, -2.950754832815939e-02, -5.193461479896214e-01, -5.062593036663570e-02, -5.062593036663569e-02, -5.179373984135549e-03, -5.120051552730679e-04, -1.210024318688909e-03, -4.772093708082557e-02, -1.798429250472250e-03, -1.798429250472245e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_gap_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.210953980741766e-08, -2.210942079317472e-08, -2.210874113897652e-08, -2.211052199903170e-08, -2.210961024505152e-08, -2.210961024505152e-08, -2.045558874139682e-05, -2.045936515110907e-05, -2.055665424874607e-05, -2.050223039285345e-05, -2.051982389432960e-05, -2.051982389432960e-05, -7.152082344527270e-03, -7.125588166265763e-03, -6.483512709306117e-03, -6.778413281346709e-03, -6.718503785200224e-03, -6.718503785200224e-03, -6.404768860573214e-01, -6.610023449544622e-01, -2.067957265493332e-03, -5.751425655651250e-01, -6.118715517926240e-01, -6.118715517926239e-01, -1.635549314575252e+02, -1.527238557060342e+02, -4.305357849574429e+00, -4.996470773169860e+02, -3.484725646781175e+02, -3.484725646781181e+02, -6.585754285498065e-06, -6.585899011824215e-06, -6.585790679712151e-06, -6.585916153327761e-06, -6.585819875712954e-06, -6.585819875712954e-06, -1.534834845106941e-04, -1.548711075139382e-04, -1.504806971003249e-04, -1.518792790717774e-04, -1.556962176136007e-04, -1.556962176136007e-04, -3.313730853404558e-02, -2.846099639432038e-02, -3.768966120276917e-02, -3.971014152363063e-02, -3.243835615874791e-02, -3.243835615874791e-02, -7.835120297259661e-01, -3.090881970675645e-01, -9.132720983510968e-01, -3.130555739686266e-04, -8.240463745959711e-01, -8.240463745959711e-01, -9.811757381291314e+02, -5.875779018147977e+02, -2.449599564429424e+03, -2.029512943054072e+00, -1.180607852023095e+03, -1.180607852023093e+03, -4.182651885998534e-02, -4.241381497880888e-02, -4.224147135639723e-02, -4.207178078999238e-02, -4.215988231941385e-02, -4.215988231941385e-02, -4.661214388491551e-02, -5.725300680295513e-02, -5.827129338372614e-02, -5.569445956254979e-02, -5.746622279502511e-02, -5.746622279502511e-02, -2.280192940577064e-02, -2.237240993509868e-01, -1.966164062024065e-01, -1.605386548215073e-01, -1.932234065544567e-01, -1.932234065544569e-01, -6.939950958876991e-02, -4.324579912330547e+00, -2.553508404050715e+00, -2.430696048781795e-01, -1.548836329922269e+00, -1.548836329922271e+00, -8.199934431149890e+01, -1.392862190495236e+04, -2.697552530031121e+03, -1.678331181885828e+00, -1.529992453608882e+03, -1.529992453608887e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_gap_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_gap_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_gap", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.170572126242806e-03, 2.170510846358152e-03, 2.170276829826894e-03, 2.171185078623209e-03, 2.170703757101880e-03, 2.170703757101880e-03, 1.622203752232408e-02, 1.622362907642631e-02, 1.626750992375573e-02, 1.627467220915731e-02, 1.626457286002748e-02, 1.626457286002748e-02, 4.047983185403633e-02, 4.024888981073292e-02, 3.508810782013300e-02, 3.929714097064734e-02, 3.814452412348153e-02, 3.814452412348153e-02, 1.149105261102175e-01, 1.229833138648243e-01, 1.141365569608769e-02, 2.928642562582857e-02, 5.315392582291870e-02, 5.315392582291869e-02, 5.662320188714539e-04, 6.125133190697940e-04, 2.818500893951803e-03, 2.987146076836853e-04, 4.330034769561204e-04, 4.330034769561192e-04, 6.427721373867422e-03, 6.377817825013883e-03, 6.425012778592488e-03, 6.380980981887189e-03, 6.402627159473632e-03, 6.402627159473632e-03, 3.280138000751601e-02, 3.352910876982371e-02, 3.257899253317730e-02, 3.330896318007890e-02, 3.328987120595980e-02, 3.328987120595980e-02, 1.031139041108272e-01, 5.892243637662176e-02, 1.104419205852509e-01, 9.014656085986957e-02, 9.702326556430185e-02, 9.702326556430185e-02, 1.110453779603503e-02, 4.946277821540903e-02, 9.820048929664593e-03, 2.467159885195139e-02, 2.359458391193015e-02, 2.359458391193016e-02, 1.951808196426920e-04, 2.843671184148083e-04, 6.017313417155214e-04, 5.916266571727548e-03, 4.603270465864825e-04, 4.603270465864855e-04, 1.534567792136087e-02, 3.020801925094024e-02, 2.490167688666156e-02, 2.057003046105828e-02, 2.273100881617062e-02, 2.273100881617062e-02, 1.734210752142825e-02, 1.478471498141974e-01, 1.200526176157203e-01, 7.758672203952532e-02, 1.001232451908853e-01, 1.001232451908853e-01, 6.372074635997534e-02, 8.188097904350863e-02, 1.204199044570824e-01, 1.654605157270820e-01, 1.615331444870067e-01, 1.615331444870069e-01, 1.491512933907464e-01, 2.625717359373922e-03, 3.760180739438177e-03, 1.910293807301413e-01, 1.085531420435873e-02, 1.085531420435873e-02, 6.141218588658773e-04, 6.428157879733731e-05, 2.803548366814294e-04, 1.021308426953776e-02, 5.020965787947811e-04, 5.020965787947791e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05