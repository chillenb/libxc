
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lag_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.997160092820895e+01, -1.997164437717508e+01, -1.997188118662974e+01, -1.997116884044420e+01, -1.997162348249519e+01, -1.997162348249519e+01, -3.264223067670724e+00, -3.264222035613833e+00, -3.264328405722096e+00, -3.264837441393509e+00, -3.264236441980581e+00, -3.264236441980581e+00, -6.338969976908654e-01, -6.334731387029574e-01, -6.237222233577721e-01, -6.281913274936097e-01, -6.337411090204285e-01, -6.337411090204285e-01, -1.842557075051162e-01, -1.855910728321393e-01, -7.532773921201671e-01, -1.522586367682963e-01, -1.846373492396561e-01, -1.846373492396561e-01, -4.339267479715980e-02, -4.396544827164139e-02, -8.499654379203128e-02, -3.868856669113676e-02, -4.353642568087189e-02, -4.353642568087189e-02, -4.853896275767147e+00, -4.854515583184532e+00, -4.853962930868236e+00, -4.854444419238432e+00, -4.854209018139844e+00, -4.854209018139844e+00, -1.880067870455189e+00, -1.891464017432845e+00, -1.877362400713857e+00, -1.886217383118626e+00, -1.891392931362843e+00, -1.891392931362843e+00, -5.435649899128888e-01, -5.859207507769161e-01, -5.162230351100432e-01, -5.345109138741321e-01, -5.635909483125602e-01, -5.635909483125602e-01, -1.316084507958055e-01, -1.988958458403671e-01, -1.283259568353576e-01, -1.782198383490053e+00, -1.381751977004451e-01, -1.381751977004451e-01, -3.795648041146142e-02, -3.901655075734894e-02, -2.983388658356428e-02, -9.876486943839363e-02, -3.542565224988460e-02, -3.542565224988461e-02, -5.558910514706655e-01, -5.534840688921137e-01, -5.543261394008944e-01, -5.549899211291239e-01, -5.546558624546754e-01, -5.546558624546754e-01, -5.379819604479334e-01, -4.754221674547839e-01, -4.927416506914992e-01, -5.094205532128699e-01, -5.008098738038643e-01, -5.008098738038643e-01, -6.144554848490449e-01, -2.377634629688251e-01, -2.715014536604220e-01, -3.333633937518801e-01, -2.996412699483396e-01, -2.996412699483395e-01, -4.295880515577847e-01, -8.418672075465559e-02, -9.618781484576039e-02, -3.123317768668729e-01, -1.115502871807643e-01, -1.115502871807642e-01, -4.782596132187185e-02, -2.779389720518498e-02, -3.176251381939556e-02, -1.061634866803535e-01, -3.176314533125927e-02, -3.176314533125926e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lag_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.610977954322230e+01, -2.610985343807019e+01, -2.611021456373039e+01, -2.610900386919302e+01, -2.610981824985624e+01, -2.610981824985624e+01, -4.224416439364273e+00, -4.224441478328551e+00, -4.225302933657989e+00, -4.224713212987779e+00, -4.224447313364574e+00, -4.224447313364574e+00, -7.935107418552914e-01, -7.922609021803593e-01, -7.579177273029815e-01, -7.646825692401440e-01, -7.930572858023404e-01, -7.930572858023404e-01, -1.994617392586065e-01, -2.036517644033294e-01, -9.636769351186263e-01, -1.295770870353733e-01, -2.007380179491319e-01, -2.007380179491319e-01, -1.702120532242872e-02, -1.761218899483789e-02, -5.370841633809442e-02, -1.064849046903513e-02, -1.743292029545730e-02, -1.743292029545730e-02, -6.387914033019999e+00, -6.389566822306393e+00, -6.388085629134079e+00, -6.389370750043195e+00, -6.388758375645315e+00, -6.388758375645315e+00, -2.287971576912256e+00, -2.307449442950081e+00, -2.276678778861807e+00, -2.291930156497170e+00, -2.315653454356432e+00, -2.315653454356432e+00, -7.074457516126279e-01, -7.780289190269540e-01, -6.703971847235605e-01, -7.074927157305372e-01, -7.364152028542096e-01, -7.364152028542096e-01, -9.619392582932888e-02, -1.869511527962379e-01, -9.446673419414162e-02, -2.366438692813681e+00, -1.086537997268124e-01, -1.086537997268124e-01, -1.035900261989999e-02, -1.124262517930721e-02, -8.507253925405739e-03, -6.710519297340919e-02, -1.022673279778935e-02, -1.022673279778936e-02, -7.405925242873186e-01, -7.354273192808063e-01, -7.373231430842782e-01, -7.387558266648309e-01, -7.380426761654010e-01, -7.380426761654010e-01, -7.169721836558663e-01, -6.038914857407360e-01, -6.385762157928840e-01, -6.696859370623056e-01, -6.540483203409981e-01, -6.540483203409982e-01, -8.155082476610780e-01, -2.471669174838022e-01, -3.083025104544615e-01, -4.178741433237542e-01, -3.607401256358615e-01, -3.607401256358615e-01, -5.431224967537890e-01, -5.144494239212329e-02, -6.351392444636847e-02, -4.000267218171648e-01, -7.980716420270846e-02, -7.980716420270847e-02, -1.950519712589697e-02, -5.400808229527566e-03, -7.408579205866862e-03, -7.603473957578356e-02, -8.917892906205619e-03, -8.917892906205612e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lag_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lag", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.314171685806212e-09, -1.314149271594191e-09, -1.314053845027682e-09, -1.314420803802900e-09, -1.314159827129046e-09, -1.314159827129046e-09, -2.046696901821765e-06, -2.046589079887702e-06, -2.043287072307777e-06, -2.047238121358211e-06, -2.046606573417979e-06, -2.046606573417979e-06, -1.757936860288575e-03, -1.769689106518125e-03, -2.095851396347723e-03, -2.025363084095306e-03, -1.762197030555480e-03, -1.762197030555480e-03, -3.597598495997492e-01, -3.408901873012103e-01, -7.860392523074513e-04, -1.104022931057353e+00, -3.540132045049508e-01, -3.540132045049508e-01, -1.823222184953560e+03, -1.596235816671311e+03, -2.375681990832800e+01, -1.386846880774122e+04, -1.662776484858915e+03, -1.662776484858915e+03, -3.400651648128040e-07, -3.390456515150209e-07, -3.399620069393352e-07, -3.391692796895417e-07, -3.395411245454983e-07, -3.395411245454983e-07, -2.526631473903112e-05, -2.446621216357945e-05, -2.569961198814757e-05, -2.506262221988925e-05, -2.417494727087328e-05, -2.417494727087328e-05, -2.522448458273664e-03, -1.208193410050842e-03, -3.169809637150483e-03, -2.006118041240180e-03, -2.086246174917902e-03, -2.086246174917902e-03, -2.535017592939687e+00, -3.303928513059590e-01, -2.761157611652510e+00, -1.414817190379942e-05, -1.826009705045532e+00, -1.826009705045532e+00, -1.564336067580625e+04, -1.063272763340826e+04, -3.276011215732834e+04, -1.002150369861937e+01, -1.550311249907948e+04, -1.550311249907947e+04, -1.008577704911762e-03, -1.456576811024411e-03, -1.326738390859689e-03, -1.208358497150506e-03, -1.270047614693824e-03, -1.270047614693825e-03, -1.011370416995477e-03, -5.169040761264058e-03, -3.893478232218419e-03, -2.859156713774878e-03, -3.367106363088483e-03, -3.367106363088482e-03, -1.027254401840560e-03, -1.390712833542341e-01, -6.950438342607304e-02, -2.284325258561408e-02, -4.049521424631039e-02, -4.049521424631038e-02, -7.950111390642290e-03, -2.764630891523086e+01, -1.239425034595631e+01, -2.639845675397436e-02, -5.177644388337863e+00, -5.177644388337862e+00, -1.065211935386046e+03, -5.268584946257028e+05, -7.873910260931760e+04, -6.292979550835841e+00, -2.756847220635970e+04, -2.756847220635980e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05