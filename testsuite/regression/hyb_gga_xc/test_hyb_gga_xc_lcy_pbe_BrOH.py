
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lcy_pbe_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.059961334252647e+01, -2.059963867739612e+01, -2.059982598297119e+01, -2.059940963782664e+01, -2.059962608235690e+01, -2.059962608235690e+01, -3.142681690137904e+00, -3.142662117809014e+00, -3.142266370855037e+00, -3.143678034888028e+00, -3.142686274108017e+00, -3.142686274108017e+00, -4.324850699771090e-01, -4.319894052909275e-01, -4.198863352318598e-01, -4.243644910859112e-01, -4.323035371211196e-01, -4.323035371211196e-01, -4.872758819466856e-02, -5.049174126731143e-02, -5.493943910785550e-01, -2.146470115835744e-02, -4.925982556226176e-02, -4.925982556226176e-02, -1.180731383547212e-05, -1.357761324468452e-05, -9.049451636479063e-04, -1.294809545685892e-06, -1.316145783161064e-05, -1.316145783161064e-05, -4.724818753591661e+00, -4.724579530732837e+00, -4.724799701720576e+00, -4.724613587924024e+00, -4.724688046867318e+00, -4.724688046867318e+00, -1.745347283726722e+00, -1.756280510876165e+00, -1.744489495053883e+00, -1.753009234864246e+00, -1.753878374730867e+00, -1.753878374730867e+00, -3.550380633721267e-01, -4.023955834929726e-01, -3.298551111617097e-01, -3.538867889822924e-01, -3.743506188858296e-01, -3.743506188858296e-01, -9.241202029456812e-03, -4.596012357592057e-02, -9.000298416770695e-03, -1.563108402706010e+00, -1.398087290076508e-02, -1.398087290076508e-02, -1.162461574936828e-06, -1.735916462457159e-06, -7.327514076975219e-07, -2.426416304151447e-03, -1.314284516320487e-06, -1.314284516320487e-06, -3.789529272232740e-01, -3.737301714280079e-01, -3.755214422587773e-01, -3.769645456822337e-01, -3.762358282760897e-01, -3.762358282760897e-01, -3.634769965184400e-01, -2.884096681804664e-01, -3.083957137768598e-01, -3.280537596223704e-01, -3.179029524549860e-01, -3.179029524549860e-01, -4.279342483004968e-01, -7.513663373407607e-02, -1.060256509257068e-01, -1.665037100718037e-01, -1.338434533705218e-01, -1.338434533705217e-01, -2.474827031570876e-01, -7.162176682183405e-04, -1.760605187994699e-03, -1.552354687819966e-01, -5.287635178398887e-03, -5.287635178398886e-03, -1.898514706219699e-05, -3.555678366675724e-08, -2.439416453486341e-07, -4.691487154402958e-03, -8.072136872716806e-07, -8.072136872695396e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lcy_pbe_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.494811109771468e+01, -2.494819782718872e+01, -2.494859251416292e+01, -2.494717214473759e+01, -2.494815677095255e+01, -2.494815677095255e+01, -3.854486690959146e+00, -3.854523587214867e+00, -3.855704044316757e+00, -3.854524377016652e+00, -3.854522824537469e+00, -3.854522824537469e+00, -6.056960317429569e-01, -6.046255365406046e-01, -5.769201605375539e-01, -5.828153876304885e-01, -6.053070184258712e-01, -6.053070184258712e-01, -1.005400206311314e-01, -1.039398647728722e-01, -7.612704414684361e-01, -4.601618489719857e-02, -1.015796200674007e-01, -1.015796200674007e-01, -2.415731431704071e-05, -2.782514802344184e-05, -1.923316374536185e-03, -2.603125156730391e-06, -2.698766796582828e-05, -2.698766796582828e-05, -6.006293680278523e+00, -6.008923301147020e+00, -6.006562546366650e+00, -6.008607217046817e+00, -6.007641963250395e+00, -6.007641963250395e+00, -2.014198737648107e+00, -2.030906835153529e+00, -2.006168875396066e+00, -2.019113810389995e+00, -2.036249865573601e+00, -2.036249865573601e+00, -5.265886269253792e-01, -5.883604450009368e-01, -4.929823189032395e-01, -5.241169252979109e-01, -5.529862642146954e-01, -5.529862642146954e-01, -1.969592819638764e-02, -8.878077863231233e-02, -1.947634234695523e-02, -2.135059286710523e+00, -3.043913881542740e-02, -3.043913881542740e-02, -2.337083767270179e-06, -3.494349499398391e-06, -1.482829022799916e-06, -5.290222921004616e-03, -2.650439153907798e-06, -2.650439153904716e-06, -5.491617153792629e-01, -5.478324389012748e-01, -5.485499671689220e-01, -5.489300180920752e-01, -5.487599155902030e-01, -5.487599155902030e-01, -5.262478089987808e-01, -4.341780737903705e-01, -4.643372232993640e-01, -4.911622220266688e-01, -4.777761432132898e-01, -4.777761432132898e-01, -6.240427970969776e-01, -1.363488660039288e-01, -1.848683288488066e-01, -2.733009308377475e-01, -2.268721338484695e-01, -2.268721338484694e-01, -3.807188733521738e-01, -1.499736862817870e-03, -3.754826353183575e-03, -2.571051155382906e-01, -1.180500607540962e-02, -1.180500607540962e-02, -3.879103143660316e-05, -7.120467019913250e-08, -4.895040104124129e-07, -1.071350560163611e-02, -1.629574694624113e-06, -1.629574694620377e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lcy_pbe_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.660254693582893e-09, -6.660210670082464e-09, -6.659938823631704e-09, -6.660661217501301e-09, -6.660232107794765e-09, -6.660232107794765e-09, -6.955759473605948e-06, -6.955775513977429e-06, -6.955146735817022e-06, -6.950795579967759e-06, -6.955658913037548e-06, -6.955658913037548e-06, -9.770128235556130e-04, -9.879112811807841e-04, -1.241375469953570e-03, -1.218545819451799e-03, -9.809522437627519e-04, -9.809522437627519e-04, 1.646584775874802e-01, 1.718783752069070e-01, -6.208759605543508e-04, 1.257927338594334e-01, 1.670892926829728e-01, 1.670892926829728e-01, 2.948067661232481e-02, 3.176527140433738e-02, 6.435158006664310e-02, 5.520910980965555e-03, 3.310892170643900e-02, 3.310892170643900e-02, -1.573858291010357e-06, -1.572293735775461e-06, -1.573697136405047e-06, -1.572480840295900e-06, -1.573061738574351e-06, -1.573061738574351e-06, -4.807214230231311e-05, -4.720151820943941e-05, -4.809229810246663e-05, -4.741946204460021e-05, -4.742750804324164e-05, -4.742750804324164e-05, 1.156449341213473e-03, 3.438834612143556e-03, 1.658448183791753e-03, 4.435367307261937e-03, 1.126911540352053e-03, 1.126911540352053e-03, 8.399857962559282e-02, 4.507659373575080e-02, 1.047727557891998e-01, -2.900406139581048e-05, 1.285930106766031e-01, 1.285930106766031e-01, 5.689708421424095e-03, 7.087051517431063e-03, 2.145348017241064e-02, 1.021111178006343e-01, 1.094756389209922e-02, 1.094756389237827e-02, 6.531808516645697e-03, 4.991518749460438e-03, 5.490695157552367e-03, 5.915123704858916e-03, 5.698467379630107e-03, 5.698467379630107e-03, 8.012465806198859e-03, 9.625206835369056e-04, 2.258111776780431e-03, 3.859953080965319e-03, 3.012396547312134e-03, 3.012396547312137e-03, 2.486598900747773e-03, 2.575931937028167e-02, 2.088518694258504e-02, 1.715910897738454e-02, 1.980066155056137e-02, 1.980066155056139e-02, 2.734347201557916e-03, 4.442759105049697e-02, 6.836856429097271e-02, 3.688958116972454e-02, 1.476647334941514e-01, 1.476647334941505e-01, 2.437101640374061e-02, 1.779835856036926e-03, 4.326565888986151e-03, 1.929495212899225e-01, 1.480897714908474e-02, 1.480897714825681e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05